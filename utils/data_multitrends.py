import hashlib
import os
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image, ImageFile
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import DataLoader, Dataset
from torchvision import models
from torchvision.transforms import Compose, Normalize, Resize, ToTensor
from tqdm import tqdm

ImageFile.LOAD_TRUNCATED_IMAGES = True


class LazyDataset(Dataset):
    def __init__(
        self,
        item_sales,
        categories,
        colors,
        fabrics,
        temporal_features,
        gtrends,
        img_paths,
        img_root,
        retrieval_curves=None,
        retrieval_available=None,
    ):
        self.item_sales = item_sales
        self.categories = categories
        self.colors = colors
        self.fabrics = fabrics
        self.temporal_features = temporal_features
        self.gtrends = gtrends
        self.img_paths = img_paths
        self.img_root = img_root
        self.retrieval_curves = retrieval_curves
        self.retrieval_available = retrieval_available
        self.transforms = Compose(
            [
                Resize((256, 256)),
                ToTensor(),
                Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]
        )

    def __len__(self):
        return len(self.item_sales)

    def __getitem__(self, idx):
        img_path = self.img_paths[idx]
        img = Image.open(os.path.join(self.img_root, img_path)).convert("RGB")
        img_tensor = self.transforms(img)

        items = [
            self.item_sales[idx],
            self.categories[idx],
            self.colors[idx],
            self.fabrics[idx],
            self.temporal_features[idx],
            self.gtrends[idx],
            img_tensor,
        ]

        if self.retrieval_curves is not None and self.retrieval_available is not None:
            items.append(self.retrieval_curves[idx])
            items.append(self.retrieval_available[idx])

        return tuple(items)


class RetrievalFeatureEncoder:
    def __init__(self, img_root, cat_dict, col_dict, fab_dict, batch_size=32):
        self.img_root = Path(img_root)
        self.cat_dict = cat_dict
        self.col_dict = col_dict
        self.fab_dict = fab_dict
        self.batch_size = batch_size
        self.transforms = Compose(
            [
                Resize((256, 256)),
                ToTensor(),
                Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]
        )
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.image_backbone = self._build_image_backbone().to(self.device)
        self.image_backbone.eval()

    @staticmethod
    def _l2_normalize(x, eps=1e-8):
        denom = np.linalg.norm(x, axis=1, keepdims=True)
        denom = np.maximum(denom, eps)
        return x / denom

    @staticmethod
    def _safe_stack(rows):
        if not rows:
            return np.empty((0, 0), dtype=np.float32)
        return np.stack(rows).astype(np.float32)

    def _build_image_backbone(self):
        # Keep the same frozen ResNet family as the GTM image branch.
        weights = None
        try:
            weights = models.ResNet50_Weights.DEFAULT
            resnet = models.resnet50(weights=weights)
        except AttributeError:
            resnet = models.resnet50(pretrained=True)
        modules = list(resnet.children())[:-1]
        backbone = torch.nn.Sequential(*modules)
        for param in backbone.parameters():
            param.requires_grad = False
        return backbone

    def _encode_images(self, image_paths):
        image_features = []
        for start in range(0, len(image_paths), self.batch_size):
            batch_paths = image_paths[start : start + self.batch_size]
            batch_images = []
            for img_path in batch_paths:
                img = Image.open(self.img_root / img_path).convert("RGB")
                batch_images.append(self.transforms(img))
            batch_tensor = torch.stack(batch_images).to(self.device)
            with torch.no_grad():
                feats = self.image_backbone(batch_tensor).flatten(1).cpu().numpy()
            image_features.append(feats)
        return np.concatenate(image_features, axis=0).astype(np.float32)

    def encode(self, df, trend_matrices, temporal_features):
        image_paths = df["image_path"].tolist()
        image_features = self._encode_images(image_paths)

        cat_ids = np.array([self.cat_dict[val] for val in df["category"].values], dtype=np.int64)
        col_ids = np.array([self.col_dict[val] for val in df["color"].values], dtype=np.int64)
        fab_ids = np.array([self.fab_dict[val] for val in df["fabric"].values], dtype=np.int64)

        cat_onehot = np.eye(len(self.cat_dict), dtype=np.float32)[cat_ids]
        col_onehot = np.eye(len(self.col_dict), dtype=np.float32)[col_ids]
        fab_onehot = np.eye(len(self.fab_dict), dtype=np.float32)[fab_ids]
        metadata_features = np.concatenate([cat_onehot, col_onehot, fab_onehot], axis=1)

        trend_features = trend_matrices.reshape(len(df), -1).astype(np.float32)

        temporal_features = temporal_features.astype(np.float32)
        temporal_scale = np.array([7.0, 53.0, 12.0, 2100.0], dtype=np.float32)
        temporal_features = temporal_features / temporal_scale

        image_features = self._l2_normalize(image_features)
        metadata_features = self._l2_normalize(metadata_features)
        trend_features = self._l2_normalize(trend_features)
        temporal_features = self._l2_normalize(temporal_features)

        # Weight blocks before the final normalization so that large blocks do not dominate by dimensionality alone.
        image_weight = 1.0
        metadata_weight = 1.0
        trend_weight = 1.0
        temporal_weight = 0.5

        combined = np.concatenate(
            [
                image_weight * image_features,
                metadata_weight * metadata_features,
                trend_weight * trend_features,
                temporal_weight * temporal_features,
            ],
            axis=1,
        )
        return self._l2_normalize(combined).astype(np.float32)


class ZeroShotDataset:
    def __init__(
        self,
        data_df,
        img_root,
        gtrends,
        cat_dict,
        col_dict,
        fab_dict,
        trend_len,
        retrieval_bank_df=None,
        use_retrieval=False,
        retrieval_top_k=5,
        retrieval_min_similarity=0.0,
        retrieval_observability_weeks=12,
        retrieval_cache_dir=None,
        retrieval_batch_size=32,
    ):
        self.data_df = data_df.copy()
        self.gtrends = gtrends
        self.cat_dict = cat_dict
        self.col_dict = col_dict
        self.fab_dict = fab_dict
        self.trend_len = trend_len
        self.img_root = Path(img_root)
        self.use_retrieval = bool(use_retrieval)
        self.retrieval_bank_df = None if retrieval_bank_df is None else retrieval_bank_df.copy()
        self.retrieval_top_k = int(retrieval_top_k)
        self.retrieval_min_similarity = float(retrieval_min_similarity)
        self.retrieval_observability_weeks = int(retrieval_observability_weeks)
        self.retrieval_cache_dir = None if retrieval_cache_dir is None else Path(retrieval_cache_dir)
        self.retrieval_batch_size = int(retrieval_batch_size)

    @staticmethod
    def _build_cache_key(df, trend_len):
        payload = "|".join(df["external_code"].astype(str).tolist()) + f"|trend_len={trend_len}"
        return hashlib.md5(payload.encode("utf-8")).hexdigest()

    def _extract_multitrends(self, df):
        gtrend_rows = []
        for _, row in tqdm(df.iterrows(), total=len(df), ascii=True):
            cat, col, fab = row["category"], row["color"], row["fabric"]
            start_date = row["release_date"]

            gtrend_start = start_date - pd.DateOffset(weeks=52)
            cat_gtrend = self.gtrends.loc[gtrend_start:start_date][cat][-52:].values[: self.trend_len]
            col_gtrend = self.gtrends.loc[gtrend_start:start_date][col][-52:].values[: self.trend_len]
            fab_gtrend = self.gtrends.loc[gtrend_start:start_date][fab][-52:].values[: self.trend_len]

            cat_gtrend = MinMaxScaler().fit_transform(cat_gtrend.reshape(-1, 1)).flatten()
            col_gtrend = MinMaxScaler().fit_transform(col_gtrend.reshape(-1, 1)).flatten()
            fab_gtrend = MinMaxScaler().fit_transform(fab_gtrend.reshape(-1, 1)).flatten()
            multitrends = np.vstack([cat_gtrend, col_gtrend, fab_gtrend])
            gtrend_rows.append(multitrends)

        return np.array(gtrend_rows, dtype=np.float32)

    def _prepare_structured_tensors(self, df, gtrend_rows):
        clean_df = df.copy()
        image_paths = clean_df["image_path"].tolist()

        clean_df = clean_df.drop(["external_code", "season", "release_date", "image_path"], axis=1)

        item_sales = torch.FloatTensor(clean_df.iloc[:, :12].values)
        temporal_features = torch.FloatTensor(clean_df.iloc[:, 13:17].values)
        categories = torch.LongTensor([self.cat_dict[val] for val in clean_df["category"].values])
        colors = torch.LongTensor([self.col_dict[val] for val in clean_df["color"].values])
        fabrics = torch.LongTensor([self.fab_dict[val] for val in clean_df["fabric"].values])
        gtrends = torch.FloatTensor(gtrend_rows)

        return item_sales, categories, colors, fabrics, temporal_features, gtrends, image_paths

    def _load_or_compute_retrieval_embeddings(self, df, trend_rows, temporal_features_np, cache_prefix):
        cache_path = None
        if self.retrieval_cache_dir is not None:
            self.retrieval_cache_dir.mkdir(parents=True, exist_ok=True)
            cache_key = self._build_cache_key(df, self.trend_len)
            cache_path = self.retrieval_cache_dir / f"{cache_prefix}_{cache_key}.npz"

        if cache_path is not None and cache_path.exists():
            cached = np.load(cache_path)
            return cached["embeddings"].astype(np.float32)

        encoder = RetrievalFeatureEncoder(
            img_root=self.img_root,
            cat_dict=self.cat_dict,
            col_dict=self.col_dict,
            fab_dict=self.fab_dict,
            batch_size=self.retrieval_batch_size,
        )
        embeddings = encoder.encode(df, trend_rows, temporal_features_np)

        if cache_path is not None:
            np.savez_compressed(cache_path, embeddings=embeddings)

        return embeddings.astype(np.float32)

    def _build_retrieval_targets(self, query_df, bank_df, query_gtrends, bank_gtrends):
        if bank_df is None or len(bank_df) == 0:
            retrieval_curves = torch.zeros((len(query_df), 12), dtype=torch.float32)
            retrieval_available = torch.zeros(len(query_df), dtype=torch.bool)
            return retrieval_curves, retrieval_available

        query_temporal_np = query_df.drop(["external_code", "season", "release_date", "image_path"], axis=1).iloc[:, 13:17].values
        bank_temporal_np = bank_df.drop(["external_code", "season", "release_date", "image_path"], axis=1).iloc[:, 13:17].values

        query_embeddings = self._load_or_compute_retrieval_embeddings(query_df, query_gtrends, query_temporal_np, "query")
        bank_embeddings = self._load_or_compute_retrieval_embeddings(bank_df, bank_gtrends, bank_temporal_np, "bank")

        bank_sales = bank_df.drop(["external_code", "season", "release_date", "image_path"], axis=1).iloc[:, :12].values.astype(np.float32)
        bank_release_dates = pd.to_datetime(bank_df["release_date"]).to_numpy(dtype="datetime64[ns]")
        bank_observed_dates = bank_release_dates + np.timedelta64(self.retrieval_observability_weeks * 7, "D")
        bank_codes = bank_df["external_code"].astype(str).values

        retrieval_curves = np.zeros((len(query_df), 12), dtype=np.float32)
        retrieval_available = np.zeros(len(query_df), dtype=bool)

        query_release_dates = pd.to_datetime(query_df["release_date"]).to_numpy(dtype="datetime64[ns]")
        query_codes = query_df["external_code"].astype(str).values

        for idx in range(len(query_df)):
            valid_mask = bank_observed_dates <= query_release_dates[idx]
            valid_mask &= bank_codes != query_codes[idx]
            valid_indices = np.where(valid_mask)[0]

            if len(valid_indices) == 0:
                continue

            sims = bank_embeddings[valid_indices] @ query_embeddings[idx]
            sims = sims.astype(np.float32)

            threshold_mask = sims >= self.retrieval_min_similarity
            valid_indices = valid_indices[threshold_mask]
            sims = sims[threshold_mask]

            if len(valid_indices) == 0:
                continue

            order = np.argsort(-sims)
            top_count = min(self.retrieval_top_k, len(order))
            chosen = order[:top_count]
            chosen_indices = valid_indices[chosen]
            chosen_sims = np.maximum(sims[chosen], 0.0)

            if float(chosen_sims.sum()) <= 1e-8:
                weights = np.ones_like(chosen_sims) / len(chosen_sims)
            else:
                weights = chosen_sims / chosen_sims.sum()

            retrieval_curves[idx] = (bank_sales[chosen_indices] * weights[:, None]).sum(axis=0)
            retrieval_available[idx] = True

        return torch.FloatTensor(retrieval_curves), torch.BoolTensor(retrieval_available)

    def preprocess_data(self):
        data = self.data_df.copy()
        print("Preparing main dataset trends...")
        query_gtrends = self._extract_multitrends(data)
        item_sales, categories, colors, fabrics, temporal_features, gtrends, image_paths = self._prepare_structured_tensors(
            data, query_gtrends
        )

        retrieval_curves = None
        retrieval_available = None
        if self.use_retrieval:
            if self.retrieval_bank_df is None:
                raise ValueError("use_retrieval=True requires a retrieval_bank_df.")
            bank_df = self.retrieval_bank_df.copy()
            print("Preparing retrieval bank trends...")
            bank_gtrends = self._extract_multitrends(bank_df)
            print("Building causal retrieval targets...")
            retrieval_curves, retrieval_available = self._build_retrieval_targets(
                query_df=data,
                bank_df=bank_df,
                query_gtrends=query_gtrends,
                bank_gtrends=bank_gtrends,
            )

        return LazyDataset(
            item_sales,
            categories,
            colors,
            fabrics,
            temporal_features,
            gtrends,
            image_paths,
            self.img_root,
            retrieval_curves=retrieval_curves,
            retrieval_available=retrieval_available,
        )

    def get_loader(self, batch_size, train=True):
        print("Starting dataset creation process...")
        data_with_gtrends = self.preprocess_data()
        if train:
            return DataLoader(data_with_gtrends, batch_size=batch_size, shuffle=True, num_workers=2)
        return DataLoader(data_with_gtrends, batch_size=1, shuffle=False, num_workers=2)
