# report_generator.py

import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import get_peft_model, LoraConfig, TaskType
from typing import List

class MedicalReportGenerator(nn.Module):
    def __init__(self, image_embedding_dim=512):
        super().__init__()
        # Use BioGPT as the base model
        self.base_model_name = 'microsoft/biogpt'
        self.tokenizer = AutoTokenizer.from_pretrained(self.base_model_name)
        self.model = AutoModelForCausalLM.from_pretrained(self.base_model_name)
        self.model.gradient_checkpointing_enable()  # Enable gradient checkpointing during training

        # PEFT configuration with target_modules specified
        peft_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            inference_mode=False,
            r=16,
            lora_alpha=32,
            lora_dropout=0.1,
            target_modules=["q_proj", "v_proj", "k_proj", "out_proj"]
        )
        self.model = get_peft_model(self.model, peft_config)

        # Projection layer to map image embeddings to model's embedding size
        self.image_projection = nn.Linear(image_embedding_dim, self.model.config.hidden_size)

        # Token embeddings for separator token
        if 'sep_token' not in self.tokenizer.special_tokens_map:
            self.tokenizer.add_special_tokens({'sep_token': '[SEP]'})
            self.model.resize_token_embeddings(len(self.tokenizer))

        self.sep_token_id = self.tokenizer.sep_token_id

        # Ensure special tokens are set
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        if self.tokenizer.bos_token_id is None:
            self.tokenizer.bos_token = self.tokenizer.eos_token
        if self.tokenizer.eos_token_id is None:
            self.tokenizer.eos_token = '</s>'
            self.tokenizer.eos_token_id = self.tokenizer.convert_tokens_to_ids('</s>')
        self.model.config.pad_token_id = self.tokenizer.pad_token_id
        self.model.config.bos_token_id = self.tokenizer.bos_token_id
        self.model.config.eos_token_id = self.tokenizer.eos_token_id

    def forward(self, image_embeddings: torch.Tensor, prompt_input_ids: torch.Tensor, target_ids: torch.Tensor = None):
        # Project image embeddings and add sequence dimension
        projected_embeddings = self.image_projection(image_embeddings).unsqueeze(1)  # (batch_size, 1, hidden_size)

        # Get separator embedding
        sep_embedding = self.model.get_input_embeddings()(torch.tensor([self.sep_token_id], device=image_embeddings.device))
        sep_embedding = sep_embedding.unsqueeze(0).expand(image_embeddings.size(0), -1, -1)  # (batch_size, 1, hidden_size)

        # Combine image and separator embeddings
        image_and_sep_embeddings = torch.cat([projected_embeddings, sep_embedding], dim=1)  # (batch_size, 2, hidden_size)

        if target_ids is not None:
            # Concatenate prompt and target input IDs
            full_input_ids = torch.cat([prompt_input_ids, target_ids], dim=1)  # (batch_size, seq_len_prompt + seq_len_target)

            # Get embeddings for the prompt and target
            token_embeddings = self.model.get_input_embeddings()(full_input_ids)  # (batch_size, seq_len_prompt + seq_len_target, hidden_size)

            # Concatenate all embeddings
            inputs_embeds = torch.cat([image_and_sep_embeddings, token_embeddings], dim=1)  # (batch_size, total_seq_len, hidden_size)

            # Create attention mask
            attention_mask = torch.ones(inputs_embeds.size()[:2], device=inputs_embeds.device, dtype=torch.long)

            # Create labels with -100 for image, separator, and prompt tokens
            labels = torch.full((inputs_embeds.size(0), inputs_embeds.size(1)), -100, dtype=torch.long, device=inputs_embeds.device)
            labels[:, image_and_sep_embeddings.size(1) + prompt_input_ids.size(1):] = target_ids  # Only compute loss for target tokens

            # Forward pass with labels
            outputs = self.model(
                inputs_embeds=inputs_embeds,
                attention_mask=attention_mask,
                labels=labels,
                return_dict=True
            )

            return outputs.loss, outputs.logits
        else:
            # For generation
            token_embeddings = self.model.get_input_embeddings()(prompt_input_ids)  # (batch_size, seq_len_prompt, hidden_size)
            inputs_embeds = torch.cat([image_and_sep_embeddings, token_embeddings], dim=1)

            attention_mask = torch.ones(inputs_embeds.size()[:2], device=inputs_embeds.device, dtype=torch.long)

            outputs = self.model.generate(
                inputs_embeds=inputs_embeds,
                attention_mask=attention_mask,
                max_length=inputs_embeds.size(1) + 150,
                min_length=inputs_embeds.size(1) + 10,
                num_return_sequences=1,
                do_sample=True,
                top_k=50,
                top_p=0.85,
                temperature=0.8,
                no_repeat_ngram_size=3,
                length_penalty=1.0,
                repetition_penalty=1.2,
                pad_token_id=self.tokenizer.pad_token_id,
                bos_token_id=self.tokenizer.bos_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
            )

            # Decode generated tokens
            generated_texts = []
            for output in outputs:
                text = self.tokenizer.decode(output, skip_special_tokens=True)
                # Optionally, remove the prompt from the generated text
                generated_texts.append(text)

            return generated_texts
