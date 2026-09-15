from __future__ import annotations

import torch
import torch.nn as nn


def masked_softmax(logits: torch.Tensor, mask: torch.Tensor, dim: int) -> torch.Tensor:
    mask = mask.to(dtype=torch.bool, device=logits.device)
    logits = logits.masked_fill(~mask, torch.finfo(logits.dtype).min)
    weights = torch.softmax(logits, dim=dim)
    return weights.masked_fill(~mask, 0.0)


class ConformerAttentionPool(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, num_roles: int, dropout: float):
        super().__init__()
        self.input_proj = nn.Linear(input_dim, hidden_dim)
        self.role_embedding = nn.Embedding(num_roles, hidden_dim)
        self.missing_embedding = nn.Embedding(num_roles, hidden_dim)
        self.score = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

    def forward(
        self,
        conformer_embeddings: torch.Tensor,
        conformer_mask: torch.Tensor,
        role_ids: torch.Tensor,
    ) -> torch.Tensor:
        # conformer_embeddings: [batch, components, conformers, input_dim]
        x = self.input_proj(conformer_embeddings)
        role = self.role_embedding(role_ids).unsqueeze(2)
        x = x + role
        logits = self.score(x).squeeze(-1)
        has_conformer = conformer_mask.any(dim=2)
        safe_mask = conformer_mask | (~has_conformer).unsqueeze(-1)
        weights = masked_softmax(logits, safe_mask, dim=2).unsqueeze(-1)
        pooled = (x * weights).sum(dim=2)
        missing = self.missing_embedding(role_ids)
        return torch.where(has_conformer.unsqueeze(-1), pooled, missing)


class Orb3DReactionAttentionModel(nn.Module):
    """Reaction-level attention model for frozen Orb/OrbMol component embeddings."""

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 256,
        num_roles: int = 16,
        num_tasks: int = 16,
        num_layers: int = 3,
        num_heads: int = 8,
        dropout: float = 0.1,
        output_dim: int = 1,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.conformer_pool = ConformerAttentionPool(input_dim, hidden_dim, num_roles, dropout)
        self.role_embedding = nn.Embedding(num_roles, hidden_dim)
        self.task_embedding = nn.Embedding(num_tasks, hidden_dim)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, hidden_dim))
        self.pair_mlp = nn.Sequential(
            nn.LayerNorm(hidden_dim * 4),
            nn.Linear(hidden_dim * 4, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
        )
        layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim * 4,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=num_layers)
        self.head = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_dim),
        )

    def _pair_tokens(self, component_tokens: torch.Tensor, component_mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        batch, components, hidden = component_tokens.shape
        pairs = []
        pair_masks = []
        for i in range(components):
            for j in range(i + 1, components):
                left = component_tokens[:, i]
                right = component_tokens[:, j]
                pair = torch.cat([left, right, left * right, torch.abs(left - right)], dim=-1)
                pairs.append(self.pair_mlp(pair))
                pair_masks.append(component_mask[:, i] & component_mask[:, j])
        if not pairs:
            empty_tokens = component_tokens.new_zeros(batch, 0, hidden)
            empty_mask = component_mask.new_zeros(batch, 0)
            return empty_tokens, empty_mask
        return torch.stack(pairs, dim=1), torch.stack(pair_masks, dim=1)

    def forward(
        self,
        conformer_embeddings: torch.Tensor,
        conformer_mask: torch.Tensor,
        component_mask: torch.Tensor,
        role_ids: torch.Tensor,
        task_ids: torch.Tensor,
    ) -> torch.Tensor:
        return self.head(
            self.encode(
                conformer_embeddings=conformer_embeddings,
                conformer_mask=conformer_mask,
                component_mask=component_mask,
                role_ids=role_ids,
                task_ids=task_ids,
            )
        ).squeeze(-1)

    def encode(
        self,
        conformer_embeddings: torch.Tensor,
        conformer_mask: torch.Tensor,
        component_mask: torch.Tensor,
        role_ids: torch.Tensor,
        task_ids: torch.Tensor,
    ) -> torch.Tensor:
        component_tokens = self.conformer_pool(conformer_embeddings, conformer_mask, role_ids)
        component_tokens = component_tokens + self.role_embedding(role_ids)
        component_tokens = component_tokens * component_mask.unsqueeze(-1).to(component_tokens.dtype)

        pair_tokens, pair_mask = self._pair_tokens(component_tokens, component_mask)
        cls = self.cls_token.expand(component_tokens.size(0), -1, -1)
        cls = cls + self.task_embedding(task_ids).unsqueeze(1)
        tokens = torch.cat([cls, component_tokens, pair_tokens], dim=1)

        cls_mask = torch.ones(component_mask.size(0), 1, dtype=torch.bool, device=component_mask.device)
        token_mask = torch.cat([cls_mask, component_mask, pair_mask], dim=1)
        encoded = self.encoder(tokens, src_key_padding_mask=~token_mask)
        return encoded[:, 0]
