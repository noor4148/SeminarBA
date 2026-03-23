from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import torch
from PIL import Image, ImageFile
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import Compose, Normalize, Resize, ToTensor
from tqdm import tqdm

ImageFile.LOAD_TRUNCATED_IMAGES = True


@dataclass
class SimilarityMetadata:
    row_id: int
    external_code: str
    release_date: str
    split: str
    season: Optional[str]
    image_path: str


class SimilarityDataset(Dataset):
    """
    Dataset for building a shared similarity backbone.

    Compared with the original ZeroShotDataset, this class:
    - keeps metadata such as external_code, release_date and split
    - returns a stable row index for joining embeddings back to metadata
    - does not mutate the input dataframe in-place
    """

    def __init__(
        self,
        data_df: pd.DataFrame,
        img_root: str,
        gtrends: pd.DataFrame,
        cat_dict: Dict,
        col_dict: Dict,
        fab_dict: Dict,
        trend_len: int,
    ) -> None:
        self.data_df = data_df.copy().reset_index(drop=True)
        self.img_root = str(img_root)
        self.gtrends = gtrends
        self.cat_dict = cat_dict
        self.col_dict = col_dict
        self.fab_dict = fab_dict
        self.trend_len = trend_len
        self.img_transforms = Compose(
            [
                Resize((256, 256)),
                ToTensor(),
                Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]
        )

        self._prepare()

    def _prepare(self) -> None:
        df = self.data_df.copy()
        gtrends_list: List[np.ndarray] = []
        image_features: List[torch.Tensor] = []
        metadata: List[SimilarityMetadata] = []

        if "split" not in df.columns:
            df["split"] = "unknown"

        for idx, row in tqdm(df.iterrows(), total=len(df), ascii=True):
            start_date = pd.Timestamp(row["release_date"])
            cat = row["category"]
            col = row["color"]
            fab = row["fabric"]
            img_path = row["image_path"]

            # Match original preprocessing: use the previous 52 weeks up to release date.
            gtrend_start = start_date - pd.DateOffset(weeks=52)
            cat_gtrend = self.gtrends.loc[gtrend_start:start_date][cat][-52:].values[: self.trend_len]
            col_gtrend = self.gtrends.loc[gtrend_start:start_date][col][-52:].values[: self.trend_len]
            fab_gtrend = self.gtrends.loc[gtrend_start:start_date][fab][-52:].values[: self.trend_len]

            cat_gtrend = MinMaxScaler().fit_transform(cat_gtrend.reshape(-1, 1)).flatten()
            col_gtrend = MinMaxScaler().fit_transform(col_gtrend.reshape(-1, 1)).flatten()
            fab_gtrend = MinMaxScaler().fit_transform(fab_gtrend.reshape(-1, 1)).flatten()
            multitrends = np.vstack([cat_gtrend, col_gtrend, fab_gtrend])

            img = Image.open(os.path.join(self.img_root, img_path)).convert("RGB")
            image_features.append(self.img_transforms(img))
            gtrends_list.append(multitrends)

            metadata.append(
                SimilarityMetadata(
                    row_id=int(idx),
                    external_code=str(row["external_code"]),
                    release_date=start_date.strftime("%Y-%m-%d"),
                    split=str(row.get("split", "unknown")),
                    season=str(row["season"]) if "season" in row and pd.notna(row["season"]) else None,
                    image_path=str(img_path),
                )
            )

        work_df = df.drop(columns=[c for c in ["external_code", "season", "release_date", "image_path", "split"] if c in df.columns])

        self.item_sales = torch.FloatTensor(work_df.iloc[:, :12].values)
        self.temporal_features = torch.FloatTensor(work_df.iloc[:, 13:17].values)
        self.categories = torch.LongTensor([self.cat_dict[val] for val in work_df["category"].values])
        self.colors = torch.LongTensor([self.col_dict[val] for val in work_df["color"].values])
        self.fabrics = torch.LongTensor([self.fab_dict[val] for val in work_df["fabric"].values])
        self.gtrends_tensor = torch.FloatTensor(np.array(gtrends_list))
        self.images = torch.stack(image_features)
        self.row_ids = torch.LongTensor([m.row_id for m in metadata])
        self.metadata = metadata

    def __len__(self) -> int:
        return len(self.row_ids)

    def __getitem__(self, idx: int):
        return (
            self.item_sales[idx],
            self.categories[idx],
            self.colors[idx],
            self.fabrics[idx],
            self.temporal_features[idx],
            self.gtrends_tensor[idx],
            self.images[idx],
            self.row_ids[idx],
        )

    def metadata_frame(self) -> pd.DataFrame:
        return pd.DataFrame([m.__dict__ for m in self.metadata])

    def get_loader(self, batch_size: int, shuffle: bool = False, num_workers: int = 4) -> DataLoader:
        return DataLoader(self, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)
