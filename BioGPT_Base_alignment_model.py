import torch
import torch.nn as nn
from transformers import AutoModel, AutoTokenizer
from typing import List, Tuple

class ImageTextAlignmentModel(nn.Module):
    def __init__(self, image_embedding_dim: int = 512):  # Changed from 2048 to 512
        super().__init__()

        # Load BioGPT and its tokenizer
        self.text_encoder = AutoModel.from_pretrained('microsoft/biogpt')
        self.tokenizer = AutoTokenizer.from_pretrained('microsoft/biogpt')

        # Projection layers
        self.image_projection = nn.Linear(image_embedding_dim, self.text_encoder.config.hidden_size)  # Align with BioGPT
        self.text_projection = nn.Linear(self.text_encoder.config.hidden_size, self.text_encoder.config.hidden_size)

    def forward(self, image_embeddings: torch.Tensor, text: List[str]) -> Tuple[torch.Tensor, torch.Tensor]:
        # Project image embeddings to match the text embedding size
        projected_image = self.image_projection(image_embeddings)

        # Tokenize text inputs
        text_encoding = self.tokenizer(
            text,
            padding=True,
            truncation=True,
            return_tensors="pt",
            max_length=512
        )
        text_encoding = {k: v.to(image_embeddings.device) for k, v in text_encoding.items()}

        # Pass text through BioGPT encoder
        text_features = self.text_encoder(**text_encoding).last_hidden_state[:, 0, :]  # Use [CLS] token
        projected_text = self.text_projection(text_features)

        return projected_image, projected_text