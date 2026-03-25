import os
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image, ImageFile
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import Compose, Normalize, Resize, ToTensor

ImageFile.LOAD_TRUNCATED_IMAGES = True


class _BaseRetrievalDataset(Dataset):
    def __init__(self, data_df, gtrends, trend_len, sales_horizon=12):
        self.data_df = data_df.reset_index(drop=True).copy()
        self.gtrends = gtrends
        self.trend_len = trend_len
        self.sales_horizon = sales_horizon

    def __len__(self):
        return len(self.data_df)

    def _row_to_multitrends(self, row):
        start_date = row['release_date']
        cat = row['category']
        col = row['color']
        fab = row['fabric']
        gtrend_start = start_date - pd.DateOffset(weeks=52)

        cat_gtrend = self.gtrends.loc[gtrend_start:start_date][cat][-52:].values[: self.trend_len]
        col_gtrend = self.gtrends.loc[gtrend_start:start_date][col][-52:].values[: self.trend_len]
        fab_gtrend = self.gtrends.loc[gtrend_start:start_date][fab][-52:].values[: self.trend_len]

        cat_gtrend = MinMaxScaler().fit_transform(cat_gtrend.reshape(-1, 1)).flatten()
        col_gtrend = MinMaxScaler().fit_transform(col_gtrend.reshape(-1, 1)).flatten()
        fab_gtrend = MinMaxScaler().fit_transform(fab_gtrend.reshape(-1, 1)).flatten()
        return torch.FloatTensor(np.vstack([cat_gtrend, col_gtrend, fab_gtrend]))

    def _sales_tensor(self, row):
        return torch.FloatTensor(row.iloc[: self.sales_horizon].values.astype(np.float32))


class RetrievalTrendDataset(_BaseRetrievalDataset):
    def __getitem__(self, idx):
        row = self.data_df.iloc[idx]
        item_sales = self._sales_tensor(row)
        multitrends = self._row_to_multitrends(row)
        return item_sales, multitrends, str(row['external_code']), str(row['split'])

    def get_loader(self, batch_size, shuffle=False, num_workers=0):
        return DataLoader(self, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)


class ProductFeatureDataset(_BaseRetrievalDataset):
    def __init__(self, data_df, img_root, gtrends, cat_dict, col_dict, fab_dict, trend_len, sales_horizon=12):
        super().__init__(data_df=data_df, gtrends=gtrends, trend_len=trend_len, sales_horizon=sales_horizon)
        self.img_root = Path(img_root)
        self.cat_dict = cat_dict
        self.col_dict = col_dict
        self.fab_dict = fab_dict
        self.img_transforms = Compose([
            Resize((256, 256)),
            ToTensor(),
            Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

    def __getitem__(self, idx):
        row = self.data_df.iloc[idx]
        item_sales = self._sales_tensor(row)
        temporal_features = torch.tensor(
            row[['day', 'week', 'month', 'year']].to_numpy(dtype=np.float32),
            dtype=torch.float32
        )
        multitrends = self._row_to_multitrends(row)
        category = torch.tensor(self.cat_dict[row['category']], dtype=torch.long)
        color = torch.tensor(self.col_dict[row['color']], dtype=torch.long)
        fabric = torch.tensor(self.fab_dict[row['fabric']], dtype=torch.long)

        img = Image.open(self.img_root / row['image_path']).convert('RGB')
        image_tensor = self.img_transforms(img)
        return (
            item_sales,
            category,
            color,
            fabric,
            temporal_features,
            multitrends,
            image_tensor,
            str(row['external_code']),
            str(row['split']),
        )

    def get_loader(self, batch_size, shuffle=False, num_workers=0):
        return DataLoader(self, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)
