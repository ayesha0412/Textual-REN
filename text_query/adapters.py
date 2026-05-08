"""
Adapter layers bridging CLIP and REN embedding spaces.

TextRegionAdapter: Linear projection mapping CLIP text embeddings (1280-dim)
to REN region token embedding space (1024-dim) for compatibility in scoring.
"""

import torch
import torch.nn as nn


class TextRegionAdapter(nn.Module):
    """
    Maps CLIP text embeddings (1024-dim, ViT-g-14) to REN region token space (1024-dim).

    Used to score REN region crops against text queries by projecting the text
    embedding into the same space as region tokens, enabling direct cosine similarity.

    Args:
        input_dim: CLIP text embedding dimension (default: 1024 for ViT-g-14)
        output_dim: REN region token dimension (default: 1024)
    """

    def __init__(self, input_dim: int = 1024, output_dim: int = 1024):
        super().__init__()
        self.projection = nn.Linear(input_dim, output_dim, bias=False)
        # Initialize with small random weights for stable training
        nn.init.normal_(self.projection.weight, mean=0.0, std=0.02)

    def forward(self, text_embedding: torch.Tensor) -> torch.Tensor:
        """
        Project text embedding to REN space.

        Args:
            text_embedding: Shape (batch_size, 1280) or (1280,)

        Returns:
            Projected embedding: Shape (..., 1024)
        """
        return self.projection(text_embedding)

    def load_pretrained(self, checkpoint_path: str, device: str = 'cpu'):
        """Load pretrained adapter weights from checkpoint."""
        state_dict = torch.load(checkpoint_path, map_location=device)
        self.load_state_dict(state_dict)
        self.to(device)

    def save_pretrained(self, checkpoint_path: str):
        """Save adapter weights to checkpoint."""
        torch.save(self.state_dict(), checkpoint_path)


class ClipRegionScorer(nn.Module):
    """
    Score REN region tokens against CLIP text embeddings with learned projection.

    Combines TextRegionAdapter with cosine similarity for end-to-end region scoring.
    """

    def __init__(self, adapter: TextRegionAdapter = None, temperature: float = 0.1):
        super().__init__()
        self.adapter = adapter or TextRegionAdapter()
        self.temperature = temperature

    def forward(
        self,
        text_embedding: torch.Tensor,
        region_tokens: torch.Tensor
    ) -> torch.Tensor:
        """
        Score region tokens against text query.

        Args:
            text_embedding: Shape (1024,) or (batch_size, 1024)
            region_tokens: Shape (num_regions, 1024) or (batch_size, num_regions, 1024)

        Returns:
            Similarity scores: Shape (num_regions,) or (batch_size, num_regions)
        """
        # Ensure 2D for adapter
        text_was_1d = text_embedding.ndim == 1
        if text_was_1d:
            text_embedding = text_embedding.unsqueeze(0)

        # Project text to REN space
        projected_text = self.adapter(text_embedding)  # (batch, 1024)

        # L2 normalize for cosine similarity
        projected_text = torch.nn.functional.normalize(projected_text, p=2, dim=-1)
        region_tokens = torch.nn.functional.normalize(region_tokens, p=2, dim=-1)

        # Cosine similarity with temperature scaling
        if region_tokens.ndim == 2:
            # Single batch: (1024,) x (num_regions, 1024) -> (num_regions,)
            scores = torch.matmul(projected_text, region_tokens.t()).squeeze(0)
        else:
            # Batched: (batch, 1024) x (batch, num_regions, 1024) -> (batch, num_regions)
            scores = torch.matmul(
                projected_text.unsqueeze(1),
                region_tokens.transpose(-2, -1)
            ).squeeze(1)

        scores = scores / self.temperature

        if text_was_1d:
            scores = scores.squeeze(0)

        return scores
