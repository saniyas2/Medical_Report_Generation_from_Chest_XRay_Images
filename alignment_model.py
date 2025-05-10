# alignment_model.py

import torch
import torch.nn as nn
from transformers import AutoModel, AutoTokenizer
from typing import List, Tuple, Optional

class ImageTextAlignmentModel(nn.Module):
    def __init__(self, image_embedding_dim: int = 512, text_embedding_dim: Optional[int] = None):
        super().__init__()

        # Initialize BioGPT encoder and tokenizer
        self.text_encoder = AutoModel.from_pretrained('microsoft/biogpt')
        self.tokenizer = AutoTokenizer.from_pretrained('microsoft/biogpt')

        if text_embedding_dim is None:
            text_embedding_dim = self.text_encoder.config.hidden_size

        # Projection networks with layer normalization
        self.image_projection = nn.Sequential(
            nn.Linear(image_embedding_dim, text_embedding_dim),
            nn.LayerNorm(text_embedding_dim),
            nn.GELU(),
        )

        self.text_projection = nn.Sequential(
            nn.Linear(text_embedding_dim, text_embedding_dim),
            nn.LayerNorm(text_embedding_dim),
            nn.GELU(),
        )

        # Initialize weights
        self._init_weights()

    def _init_weights(self):
        """Initialize weights with Xavier uniform distribution"""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def encode_text(self, text: List[str], device: torch.device) -> torch.Tensor:
        """Encode text using BioGPT"""
        # Tokenize and encode text
        text_encoding = self.tokenizer(
            text,
            padding=True,
            truncation=True,
            return_tensors="pt",
            max_length=512
        ).to(device)

        # Get text features
        with torch.no_grad():
            text_outputs = self.text_encoder(**text_encoding)
            text_features = text_outputs.last_hidden_state[:, 0, :]  # Take [CLS] token

        return text_features

    def forward(self, image_embeddings: torch.Tensor, text: List[str]) -> Tuple[torch.Tensor, torch.Tensor]:
        # Get device
        device = image_embeddings.device

        # Encode text
        text_features = self.encode_text(text, device)

        # Project features
        projected_image = self.image_projection(image_embeddings)
        projected_text = self.text_projection(text_features)

        return projected_image, projected_text
