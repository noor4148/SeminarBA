import argparse
from pathlib import Path

import pandas as pd
import pytorch_lightning as pl
import torch

from models.GTM import GTM
from utils.data_retrieval import ProductFeatureDataset
from utils.embedding_store import RetrievalStore
from utils.retrieval_metadata import combine_splits


def run(args):
    print(args)
    device = torch.device(f'cuda:{args.gpu_num}' if torch.cuda.is_available() else 'cpu')
    pl.seed_everything(args.seed)

    combined_df = combine_splits(data_folder=args.data_folder, val_fraction=args.val_fraction)
    cat_dict = torch.load(Path(args.data_folder) / 'category_labels.pt', weights_only=False)
    col_dict = torch.load(Path(args.data_folder) / 'color_labels.pt', weights_only=False)
    fab_dict = torch.load(Path(args.data_folder) / 'fabric_labels.pt', weights_only=False)
    gtrends = pd.read_csv(Path(args.data_folder) / 'gtrends.csv', index_col=[0], parse_dates=True)

    dataset = ProductFeatureDataset(
        data_df=combined_df,
        img_root=Path(args.data_folder) / 'images',
        gtrends=gtrends,
        cat_dict=cat_dict,
        col_dict=col_dict,
        fab_dict=fab_dict,
        trend_len=args.trend_len,
        sales_horizon=args.sales_horizon,
    )
    loader = dataset.get_loader(batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    model = GTM(
        embedding_dim=args.embedding_dim,
        hidden_dim=args.hidden_dim,
        output_dim=args.model_output_dim,
        num_heads=args.num_attn_heads,
        num_layers=args.num_hidden_layers,
        cat_dict=cat_dict,
        col_dict=col_dict,
        fab_dict=fab_dict,
        use_text=args.use_text,
        use_img=args.use_img,
        trend_len=args.trend_len,
        num_trends=args.num_trends,
        use_encoder_mask=args.use_encoder_mask,
        autoregressive=args.autoregressive,
        gpu_num=args.gpu_num,
    )
    ckpt = torch.load(args.baseline_ckpt_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt['state_dict'], strict=True)
    model.to(device)
    model.eval()

    all_embeddings = []
    all_sales = []
    all_codes = []
    all_splits = []

    with torch.no_grad():
        for batch in loader:
            item_sales, category, color, fabric, temporal_features, _, images, item_codes, split_names = batch
            category = category.to(device)
            color = color.to(device)
            fabric = fabric.to(device)
            temporal_features = temporal_features.to(device)
            images = images.to(device)
            static_embeddings = model.encode_static_features(category, color, fabric, temporal_features, images)
            all_embeddings.append(static_embeddings.detach().cpu())
            all_sales.append(item_sales[:, : args.sales_horizon].detach().cpu())
            all_codes.extend(list(item_codes))
            all_splits.extend(list(split_names))

    embeddings = torch.cat(all_embeddings, dim=0)
    sales = torch.cat(all_sales, dim=0)

    metadata = combined_df.copy().reset_index(drop=True)
    metadata['external_code'] = all_codes
    metadata['split'] = all_splits
    metadata['store_idx'] = range(len(metadata))
    metadata = metadata[
        ['external_code', 'release_date', 'split', 'category', 'color', 'fabric', 'season', 'image_path', 'store_idx']
    ]

    store = RetrievalStore(embeddings=embeddings, sales=sales, metadata=metadata)
    store.save(args.output_path)
    print(f'Saved retrieval store to: {args.output_path}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Extract frozen GTM embeddings for retrieval training.')
    parser.add_argument('--data_folder', type=str, default='dataset/')
    parser.add_argument('--baseline_ckpt_path', type=str, required=True)
    parser.add_argument('--output_path', type=str, default='artifacts/retrieval_store.pt')
    parser.add_argument('--seed', type=int, default=21)
    parser.add_argument('--gpu_num', type=int, default=0)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--num_workers', type=int, default=0)
    parser.add_argument('--val_fraction', type=float, default=0.15)
    parser.add_argument('--sales_horizon', type=int, default=12)

    parser.add_argument('--use_img', type=int, default=1)
    parser.add_argument('--use_text', type=int, default=1)
    parser.add_argument('--trend_len', type=int, default=52)
    parser.add_argument('--num_trends', type=int, default=3)
    parser.add_argument('--embedding_dim', type=int, default=32)
    parser.add_argument('--hidden_dim', type=int, default=64)
    parser.add_argument('--model_output_dim', type=int, default=12)
    parser.add_argument('--use_encoder_mask', type=int, default=1)
    parser.add_argument('--autoregressive', type=int, default=0)
    parser.add_argument('--num_attn_heads', type=int, default=4)
    parser.add_argument('--num_hidden_layers', type=int, default=1)
    args = parser.parse_args()
    run(args)
