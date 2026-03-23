from __future__ import annotations

from typing import Dict, Optional, Tuple

import pandas as pd
import torch

from .similarity_backbone import BackboneStore


class SimilarityIndex:
    """
    Central access point for the shared similarity backbone.

    One store can be reused for:
    - historical retrieval (train-only + full horizon observed before target launch)
    - competition-aware neighborhoods (active at the same time)
    """

    def __init__(self, store: BackboneStore) -> None:
        self.store = store
        self.metadata = store.metadata.copy().reset_index(drop=True)
        self.backbone = store.normalized_backbone.float()
        self.sales = store.sales.float()
        self.release_dates = pd.to_datetime(self.metadata["release_date"])
        self.splits = self.metadata["split"].fillna("unknown").astype(str)
        self.codes = self.metadata["external_code"].astype(str)

    def __len__(self) -> int:
        return len(self.metadata)

    def similarity_row(self, item_idx: int) -> torch.Tensor:
        return self.backbone[item_idx] @ self.backbone.T

    def precompute_topk(self, k: int, batch_size: int = 512, exclude_self: bool = True) -> None:
        if k <= 0:
            raise ValueError("k must be positive.")

        all_indices = []
        all_scores = []
        n = len(self)

        for start in range(0, n, batch_size):
            end = min(start + batch_size, n)
            sims = self.backbone[start:end] @ self.backbone.T
            if exclude_self:
                row_ids = torch.arange(start, end)
                sims[torch.arange(end - start), row_ids] = float("-inf")
            scores, indices = torch.topk(sims, k=min(k, sims.shape[1]), dim=1)
            all_indices.append(indices.cpu())
            all_scores.append(scores.cpu())

        self.store.topk_indices = torch.cat(all_indices, dim=0)
        self.store.topk_scores = torch.cat(all_scores, dim=0)

    def _candidate_pool(self, item_idx: int, candidate_pool_size: Optional[int] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        if self.store.topk_indices is not None and self.store.topk_scores is not None:
            indices = self.store.topk_indices[item_idx]
            scores = self.store.topk_scores[item_idx]
            if candidate_pool_size is not None:
                indices = indices[:candidate_pool_size]
                scores = scores[:candidate_pool_size]
            return indices.clone(), scores.clone()

        sims = self.similarity_row(item_idx)
        sims[item_idx] = float("-inf")
        if candidate_pool_size is None:
            candidate_pool_size = len(self) - 1
        scores, indices = torch.topk(sims, k=min(candidate_pool_size, len(self) - 1), dim=0)
        return indices.cpu(), scores.cpu()

    def retrieval_mask(self, item_idx: int, horizon_weeks: int, train_only: bool = True) -> torch.Tensor:
        target_date = self.release_dates.iloc[item_idx]
        candidate_complete_date = self.release_dates + pd.to_timedelta(horizon_weeks, unit="W")
        mask = candidate_complete_date <= target_date
        if train_only:
            mask &= self.splits.eq("train")
        mask.iloc[item_idx] = False
        return torch.as_tensor(mask.values, dtype=torch.bool)

    def competition_mask(
        self,
        item_idx: int,
        at_date: Optional[str | pd.Timestamp] = None,
        active_weeks: int = 12,
        allow_self: bool = False,
    ) -> torch.Tensor:
        if at_date is None:
            at_date = self.release_dates.iloc[item_idx]
        at_date = pd.Timestamp(at_date)
        active_until = self.release_dates + pd.to_timedelta(active_weeks, unit="W")
        mask = (self.release_dates <= at_date) & (active_until >= at_date)
        if not allow_self:
            mask.iloc[item_idx] = False
        return torch.as_tensor(mask.values, dtype=torch.bool)

    def get_retrieval_neighbors(
        self,
        item_idx: int,
        horizon_weeks: int,
        top_k: int,
        candidate_pool_size: Optional[int] = None,
        train_only: bool = True,
    ) -> pd.DataFrame:
        candidate_idx, candidate_scores = self._candidate_pool(item_idx, candidate_pool_size)
        mask = self.retrieval_mask(item_idx, horizon_weeks=horizon_weeks, train_only=train_only)
        keep = mask[candidate_idx]
        filtered_idx = candidate_idx[keep][:top_k]
        filtered_scores = candidate_scores[keep][:top_k]
        return self._neighbor_frame(filtered_idx, filtered_scores, relation="retrieval")

    def get_competition_neighbors(
        self,
        item_idx: int,
        top_k: int,
        at_date: Optional[str | pd.Timestamp] = None,
        active_weeks: int = 12,
        candidate_pool_size: Optional[int] = None,
    ) -> pd.DataFrame:
        candidate_idx, candidate_scores = self._candidate_pool(item_idx, candidate_pool_size)
        mask = self.competition_mask(item_idx, at_date=at_date, active_weeks=active_weeks, allow_self=False)
        keep = mask[candidate_idx]
        filtered_idx = candidate_idx[keep][:top_k]
        filtered_scores = candidate_scores[keep][:top_k]
        return self._neighbor_frame(filtered_idx, filtered_scores, relation="competition")

    def _neighbor_frame(self, indices: torch.Tensor, scores: torch.Tensor, relation: str) -> pd.DataFrame:
        if len(indices) == 0:
            return pd.DataFrame(columns=["neighbor_idx", "external_code", "release_date", "split", "score", "relation"])
        out = self.metadata.iloc[indices.tolist()].copy().reset_index(drop=True)
        out.insert(0, "neighbor_idx", indices.tolist())
        out["score"] = scores.tolist()
        out["relation"] = relation
        return out

    def simple_retrieval_prior(
        self,
        item_idx: int,
        horizon_weeks: int,
        top_k: int,
        weighted: bool = True,
        candidate_pool_size: Optional[int] = None,
        train_only: bool = True,
    ) -> Dict[str, torch.Tensor | pd.DataFrame]:
        neighbors = self.get_retrieval_neighbors(
            item_idx=item_idx,
            horizon_weeks=horizon_weeks,
            top_k=top_k,
            candidate_pool_size=candidate_pool_size,
            train_only=train_only,
        )
        if neighbors.empty:
            return {
                "neighbors": neighbors,
                "prior_curve": torch.zeros(horizon_weeks, dtype=torch.float32),
            }

        idx = torch.as_tensor(neighbors["neighbor_idx"].values, dtype=torch.long)
        curves = self.sales[idx, :horizon_weeks]
        if weighted:
            weights = torch.as_tensor(neighbors["score"].values, dtype=torch.float32)
            weights = torch.softmax(weights, dim=0)
            prior_curve = (weights.unsqueeze(-1) * curves).sum(dim=0)
        else:
            prior_curve = curves.mean(dim=0)

        return {"neighbors": neighbors, "prior_curve": prior_curve}
