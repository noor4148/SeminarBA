from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
import pytorch_lightning as pl
import torch

from models.GTM import GTM
from models.FCN import FCN

from .similarity_backbone import build_backbone_store, save_backbone_store
from .similarity_dataset import SimilarityDataset
from .retrieval_index import SimilarityIndex


def load_model(args, cat_dict, col_dict, fab_dict, device: torch.device):
    if args.model_type == "FCN":
        model = FCN(
            embedding_dim=args.embedding_dim,
            hidden_dim=args.hidden_dim,
            output_dim=args.model_output_dim,
            cat_dict=cat_dict,
            col_dict=col_dict,
            fab_dict=fab_dict,
            use_trends=args.use_trends,
            use_text=args.use_text,
            use_img=args.use_img,
            trend_len=args.trend_len,
            num_trends=args.num_trends,
            use_encoder_mask=args.use_encoder_mask,
            gpu_num=args.gpu_num,
        )
    else:
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

    ckpt = torch.load(args.ckpt_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["state_dict"], strict=True)
    return model


def main(args):
    print(args)
    pl.seed_everything(args.seed)

    device = torch.device(f"cuda:{args.gpu_num}" if torch.cuda.is_available() else "cpu")

    train_df = pd.read_csv(Path(args.data_folder) / "train.csv", parse_dates=["release_date"])
    test_df = pd.read_csv(Path(args.data_folder) / "test.csv", parse_dates=["release_date"])
    train_df = train_df.copy()
    test_df = test_df.copy()
    train_df["split"] = "train"
    test_df["split"] = "test"

    full_df = pd.concat([train_df, test_df], axis=0, ignore_index=True)
    full_df = full_df.sort_values("release_date").reset_index(drop=True)

    cat_dict = torch.load(Path(args.data_folder) / "category_labels.pt")
    col_dict = torch.load(Path(args.data_folder) / "color_labels.pt")
    fab_dict = torch.load(Path(args.data_folder) / "fabric_labels.pt")
    gtrends = pd.read_csv(Path(args.data_folder) / "gtrends.csv", index_col=[0], parse_dates=True)

    dataset = SimilarityDataset(
        data_df=full_df,
        img_root=Path(args.data_folder) / "images",
        gtrends=gtrends,
        cat_dict=cat_dict,
        col_dict=col_dict,
        fab_dict=fab_dict,
        trend_len=args.trend_len,
    )
    loader = dataset.get_loader(batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    model = load_model(args, cat_dict, col_dict, fab_dict, device)

    store = build_backbone_store(
        model=model,
        loader=loader,
        metadata_df=dataset.metadata_frame(),
        device=device,
        include_trends=bool(args.include_trends),
        trend_pooling=args.trend_pooling,
        normalize_backbone=True,
    )

    if args.precompute_topk > 0:
        index = SimilarityIndex(store)
        index.precompute_topk(k=args.precompute_topk, batch_size=args.topk_batch_size, exclude_self=True)
        store = index.store

    save_backbone_store(store, args.output_path)
    print(f"Saved backbone store to: {args.output_path}")
    print(f"Items: {store.config['num_items']} | backbone_dim: {store.config['backbone_dim']}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Build shared similarity backbone for GTM extensions")

    parser.add_argument("--data_folder", type=str, default="dataset/")
    parser.add_argument("--ckpt_path", type=str, required=True)
    parser.add_argument("--output_path", type=str, default="artifacts/similarity_backbone.pt")
    parser.add_argument("--gpu_num", type=int, default=0)
    parser.add_argument("--seed", type=int, default=21)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--batch_size", type=int, default=16)

    parser.add_argument("--model_type", type=str, default="GTM", choices=["GTM", "FCN"])
    parser.add_argument("--use_trends", type=int, default=1)
    parser.add_argument("--use_img", type=int, default=1)
    parser.add_argument("--use_text", type=int, default=1)
    parser.add_argument("--trend_len", type=int, default=52)
    parser.add_argument("--num_trends", type=int, default=3)
    parser.add_argument("--embedding_dim", type=int, default=32)
    parser.add_argument("--hidden_dim", type=int, default=64)
    parser.add_argument("--model_output_dim", type=int, default=12)
    parser.add_argument("--use_encoder_mask", type=int, default=1)
    parser.add_argument("--autoregressive", type=int, default=0)
    parser.add_argument("--num_attn_heads", type=int, default=4)
    parser.add_argument("--num_hidden_layers", type=int, default=1)

    parser.add_argument("--include_trends", type=int, default=1)
    parser.add_argument("--trend_pooling", type=str, default="mean", choices=["mean", "max", "last"])
    parser.add_argument("--precompute_topk", type=int, default=200)
    parser.add_argument("--topk_batch_size", type=int, default=512)

    main(parser.parse_args())
