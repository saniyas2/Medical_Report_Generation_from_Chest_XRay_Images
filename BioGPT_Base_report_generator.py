import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import get_peft_model, LoraConfig, TaskType
from typing import List

class MedicalReportGenerator(nn.Module):
    def __init__(self):
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
        self.input_projection = nn.Linear(self.model.config.hidden_size, self.model.config.hidden_size)

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

    def forward(self, input_embeddings: torch.Tensor, target_ids: torch.Tensor = None):
        # Project input embeddings to model's hidden size
        projected_embeddings = self.input_projection(input_embeddings)
        projected_embeddings = projected_embeddings.unsqueeze(1)  # Add sequence dimension

        if target_ids is not None:
            # Get token embeddings for the target sequence
            token_embeddings = self.model.get_input_embeddings()(target_ids)
            # Concatenate projected image embeddings with token embeddings
            inputs_embeds = torch.cat([projected_embeddings, token_embeddings], dim=1)
            # Adjust attention mask
            attention_mask = torch.ones(inputs_embeds.size()[:2], device=input_embeddings.device, dtype=torch.long)
            # Pad labels with -100 at the beginning to match input length
            padding = torch.full((target_ids.size(0), 1), -100, dtype=torch.long, device=target_ids.device)
            labels = torch.cat([padding, target_ids], dim=1)
            # Forward pass with labels
            outputs = self.model(
                inputs_embeds=inputs_embeds,
                attention_mask=attention_mask,
                labels=labels,
                return_dict=True
            )
            return outputs.loss, outputs.logits
        else:
            raise ValueError("Target IDs must be provided during training.")

    def generate_report(self, input_embeddings: torch.Tensor, max_length: int = 150) -> List[str]:
        # Temporarily disable gradient checkpointing
        self.model.gradient_checkpointing_disable()
        # Project input embeddings to model's hidden size
        projected_embeddings = self.input_projection(input_embeddings)
        projected_embeddings = projected_embeddings.unsqueeze(1)  # Shape: (batch_size, 1, hidden_size)

        # Get BOS token id
        bos_token_id = self.tokenizer.bos_token_id
        if bos_token_id is None:
            raise ValueError("bos_token_id is not set in the tokenizer.")

        # Get embedding of BOS token
        bos_embedding = self.model.get_input_embeddings()(torch.tensor([[bos_token_id]]).to(input_embeddings.device))
        # Shape: (1, 1, hidden_size)

        # Repeat bos_embedding for batch size
        bos_embedding = bos_embedding.expand(input_embeddings.size(0), -1, -1)  # Shape: (batch_size, 1, hidden_size)

        # Concatenate bos_embedding and projected_embeddings
        inputs_embeds = torch.cat([bos_embedding, projected_embeddings], dim=1)  # Shape: (batch_size, 2, hidden_size)

        # Create attention mask
        batch_size = inputs_embeds.size(0)
        attention_mask = torch.ones((batch_size, inputs_embeds.size(1)), device=inputs_embeds.device, dtype=torch.long)

        # Generate text
        outputs = self.model.generate(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            max_length=max_length,
            min_length=10,
            num_return_sequences=1,
            do_sample=True,
            top_k=50,
            top_p=0.85,
            temperature=0.8,
            length_penalty=1.0,
            repetition_penalty=1.2,
            no_repeat_ngram_size=3,
            pad_token_id=self.tokenizer.pad_token_id,
            eos_token_id=self.tokenizer.eos_token_id,
            use_cache=True,
        )
        # Re-enable gradient checkpointing
        self.model.gradient_checkpointing_enable()
        generated_texts = self.tokenizer.batch_decode(outputs, skip_special_tokens=True)
        return generated_texts