from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, Literal, Optional

import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

TrendPooling = Literal["mean", "max", "last"]


@dataclass
class BackboneStore:
    metadata: pd.DataFrame
    sales: torch.Tensor
    static_embedding: torch.Tensor
    trend_embedding: torch.Tensor
    backbone_embedding: torch.Tensor
    normalized_backbone: torch.Tensor
    config: Dict
    topk_indices: Optional[torch.Tensor] = None
    topk_scores: Optional[torch.Tensor] = None


class LaunchEmbeddingExtractor:
    """
    Extracts launch-only product embeddings from the existing GTM/FCN model.

    No edits to GTM.py / FCN.py are required: this wrapper directly reuses the
    public encoders already defined on the model object.
    """

    def __init__(
        self,
        model,
        device: torch.device,
        include_trends: bool = True,
        trend_pooling: TrendPooling = "mean",
        normalize_backbone: bool = True,
    ) -> None:
        self.model = model
        self.device = device
        self.include_trends = include_trends
        self.trend_pooling = trend_pooling
        self.normalize_backbone = normalize_backbone

    def _pool_trends(self, gtrend_encoding: torch.Tensor) -> torch.Tensor:
        # Model encoder returns [trend_len, batch, hidden_dim]
        if self.trend_pooling == "mean":
            return gtrend_encoding.mean(dim=0)
        if self.trend_pooling == "max":
            return gtrend_encoding.max(dim=0).values
        if self.trend_pooling == "last":
            return gtrend_encoding[-1]
        raise ValueError(f"Unsupported trend_pooling={self.trend_pooling}")

    def encode_batch(
        self,
        category: torch.Tensor,
        color: torch.Tensor,
        fabric: torch.Tensor,
        temporal_features: torch.Tensor,
        gtrends: torch.Tensor,
        images: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        with torch.no_grad():
            img_encoding = self.model.image_encoder(images)
            dummy_encoding = self.model.dummy_encoder(temporal_features)
            text_encoding = self.model.text_encoder(category, color, fabric)
            static_embedding = self.model.static_feature_encoder(img_encoding, text_encoding, dummy_encoding)

            if self.include_trends:
                gtrend_encoding = self.model.gtrend_encoder(gtrends)
                trend_embedding = self._pool_trends(gtrend_encoding)
                backbone_embedding = torch.cat([static_embedding, trend_embedding], dim=-1)
            else:
                trend_embedding = torch.empty(static_embedding.shape[0], 0, device=static_embedding.device)
                backbone_embedding = static_embedding

            if self.normalize_backbone:
                normalized_backbone = F.normalize(backbone_embedding, p=2, dim=-1)
            else:
                normalized_backbone = backbone_embedding

        return {
            "static_embedding": static_embedding,
            "trend_embedding": trend_embedding,
            "backbone_embedding": backbone_embedding,
            "normalized_backbone": normalized_backbone,
        }


def _stack_parts(parts: Iterable[torch.Tensor]) -> torch.Tensor:
    tensors = list(parts)
    if not tensors:
        raise ValueError("Nothing to stack.")
    return torch.cat(tensors, dim=0)


def build_backbone_store(
    model,
    loader: DataLoader,
    metadata_df: pd.DataFrame,
    device: torch.device,
    include_trends: bool = True,
    trend_pooling: TrendPooling = "mean",
    normalize_backbone: bool = True,
) -> BackboneStore:
    extractor = LaunchEmbeddingExtractor(
        model=model,
        device=device,
        include_trends=include_trends,
        trend_pooling=trend_pooling,
        normalize_backbone=normalize_backbone,
    )

    model.to(device)
    model.eval()

    sales_parts = []
    static_parts = []
    trend_parts = []
    backbone_parts = []
    normalized_parts = []
    row_ids = []

    for batch in tqdm(loader, total=len(loader), ascii=True, desc="Building backbone"):
        item_sales, category, color, fabric, temporal_features, gtrends, images, batch_row_ids = batch
        item_sales = item_sales.to(device)
        category = category.to(device)
        color = color.to(device)
        fabric = fabric.to(device)
        temporal_features = temporal_features.to(device)
        gtrends = gtrends.to(device)
        images = images.to(device)

        encoded = extractor.encode_batch(category, color, fabric, temporal_features, gtrends, images)
        sales_parts.append(item_sales.detach().cpu())
        static_parts.append(encoded["static_embedding"].detach().cpu())
        trend_parts.append(encoded["trend_embedding"].detach().cpu())
        backbone_parts.append(encoded["backbone_embedding"].detach().cpu())
        normalized_parts.append(encoded["normalized_backbone"].detach().cpu())
        row_ids.extend(batch_row_ids.detach().cpu().tolist())

    order_df = pd.DataFrame({"row_id": row_ids, "_order": range(len(row_ids))})
    ordered_metadata = metadata_df.merge(order_df, on="row_id", how="inner").sort_values("_order").drop(columns="_order")
    ordered_metadata = ordered_metadata.reset_index(drop=True)

    return BackboneStore(
        metadata=ordered_metadata,
        sales=_stack_parts(sales_parts),
        static_embedding=_stack_parts(static_parts),
        trend_embedding=_stack_parts(trend_parts),
        backbone_embedding=_stack_parts(backbone_parts),
        normalized_backbone=_stack_parts(normalized_parts),
        config={
            "include_trends": include_trends,
            "trend_pooling": trend_pooling,
            "normalize_backbone": normalize_backbone,
            "num_items": len(ordered_metadata),
            "backbone_dim": _stack_parts(backbone_parts).shape[-1],
            "static_dim": _stack_parts(static_parts).shape[-1],
            "trend_dim": _stack_parts(trend_parts).shape[-1],
        },
    )


def save_backbone_store(store: BackboneStore, output_path: str | Path) -> None:
    payload = {
        "metadata": store.metadata,
        "sales": store.sales,
        "static_embedding": store.static_embedding,
        "trend_embedding": store.trend_embedding,
        "backbone_embedding": store.backbone_embedding,
        "normalized_backbone": store.normalized_backbone,
        "config": store.config,
        "topk_indices": store.topk_indices,
        "topk_scores": store.topk_scores,
    }
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, output_path)


def load_backbone_store(path: str | Path) -> BackboneStore:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    return BackboneStore(
        metadata=payload["metadata"],
        sales=payload["sales"],
        static_embedding=payload["static_embedding"],
        trend_embedding=payload["trend_embedding"],
        backbone_embedding=payload["backbone_embedding"],
        normalized_backbone=payload["normalized_backbone"],
        config=payload["config"],
        topk_indices=payload.get("topk_indices"),
        topk_scores=payload.get("topk_scores"),
    )
