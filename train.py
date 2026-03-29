import argparse
import os
from datetime import datetime
from pathlib import Path

import pandas as pd
import pytorch_lightning as pl
import torch
from pytorch_lightning import loggers as pl_loggers

from models.FCN import FCN
from models.GTM import GTM
from utils.data_multitrends import ZeroShotDataset

os.environ["TOKENIZERS_PARALLELISM"] = "false"


def run(args):
    print(args)
    pl.seed_everything(args.seed)

    train_df = pd.read_csv(Path(args.data_folder) / "train.csv", parse_dates=["release_date"])

    cat_dict = torch.load(Path(args.data_folder) / "category_labels.pt", weights_only=False)
    col_dict = torch.load(Path(args.data_folder) / "color_labels.pt", weights_only=False)
    fab_dict = torch.load(Path(args.data_folder) / "fabric_labels.pt", weights_only=False)

    gtrends = pd.read_csv(Path(args.data_folder) / "gtrends.csv", index_col=[0], parse_dates=True)

    train_df = train_df.sort_values("release_date").reset_index(drop=True)

    val_size = max(1, int(0.15 * len(train_df)))
    subtrain_df = train_df.iloc[:-val_size].copy().reset_index(drop=True)
    val_df = train_df.iloc[-val_size:].copy().reset_index(drop=True)

    retrieval_cache_dir = Path(args.retrieval_cache_dir) if args.retrieval_cache_dir else Path(args.data_folder) / "retrieval_cache"

    train_loader = ZeroShotDataset(
        subtrain_df,
        Path(args.data_folder) / "images",
        gtrends,
        cat_dict,
        col_dict,
        fab_dict,
        args.trend_len,
        retrieval_bank_df=subtrain_df if args.use_retrieval else None,
        use_retrieval=args.use_retrieval,
        retrieval_top_k=args.retrieval_top_k,
        retrieval_min_similarity=args.retrieval_min_similarity,
        retrieval_observability_weeks=args.retrieval_observability_weeks,
        retrieval_cache_dir=retrieval_cache_dir,
        retrieval_batch_size=args.retrieval_batch_size,
    ).get_loader(batch_size=args.batch_size, train=True)

    val_loader = ZeroShotDataset(
        val_df,
        Path(args.data_folder) / "images",
        gtrends,
        cat_dict,
        col_dict,
        fab_dict,
        args.trend_len,
        retrieval_bank_df=subtrain_df if args.use_retrieval else None,
        use_retrieval=args.use_retrieval,
        retrieval_top_k=args.retrieval_top_k,
        retrieval_min_similarity=args.retrieval_min_similarity,
        retrieval_observability_weeks=args.retrieval_observability_weeks,
        retrieval_cache_dir=retrieval_cache_dir,
        retrieval_batch_size=args.retrieval_batch_size,
    ).get_loader(batch_size=1, train=False)

    if args.model_type == "FCN":
        model = FCN(
            embedding_dim=args.embedding_dim,
            hidden_dim=args.hidden_dim,
            output_dim=args.output_dim,
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
            output_dim=args.output_dim,
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
            use_retrieval=args.use_retrieval,
            retrieval_seq_len=args.output_dim,
        )

    dt_string = datetime.now().strftime("%d-%m-%Y-%H-%M-%S")
    model_savename = args.model_type + "_" + args.wandb_run

    checkpoint_callback = pl.callbacks.ModelCheckpoint(
        dirpath=args.log_dir + "/" + args.model_type,
        filename=model_savename + "---{epoch}---" + dt_string,
        monitor="val_mae",
        mode="min",
        save_top_k=1,
    )

    tb_logger = pl_loggers.TensorBoardLogger(args.log_dir + "/", name=model_savename)
    trainer = pl.Trainer(
        gpus=[args.gpu_num],
        max_epochs=args.epochs,
        check_val_every_n_epoch=5,
        logger=tb_logger,
        callbacks=[checkpoint_callback],
    )

    trainer.fit(model, train_dataloaders=train_loader, val_dataloaders=val_loader)
    print(checkpoint_callback.best_model_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Zero-shot sales forecasting")

    parser.add_argument("--data_folder", type=str, default="dataset/")
    parser.add_argument("--log_dir", type=str, default="log")
    parser.add_argument("--seed", type=int, default=21)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--gpu_num", type=int, default=0)

    parser.add_argument("--model_type", type=str, default="GTM", help="Choose between GTM or FCN")
    parser.add_argument("--use_trends", type=int, default=1)
    parser.add_argument("--use_img", type=int, default=1)
    parser.add_argument("--use_text", type=int, default=1)
    parser.add_argument("--trend_len", type=int, default=52)
    parser.add_argument("--num_trends", type=int, default=3)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--embedding_dim", type=int, default=32)
    parser.add_argument("--hidden_dim", type=int, default=64)
    parser.add_argument("--output_dim", type=int, default=12)
    parser.add_argument("--use_encoder_mask", type=int, default=1)
    parser.add_argument("--autoregressive", type=int, default=0)
    parser.add_argument("--num_attn_heads", type=int, default=4)
    parser.add_argument("--num_hidden_layers", type=int, default=1)

    parser.add_argument("--use_retrieval", type=int, default=1)
    parser.add_argument("--retrieval_top_k", type=int, default=5)
    parser.add_argument("--retrieval_min_similarity", type=float, default=0.2)
    parser.add_argument("--retrieval_observability_weeks", type=int, default=12)
    parser.add_argument("--retrieval_cache_dir", type=str, default="")
    parser.add_argument("--retrieval_batch_size", type=int, default=32)

    parser.add_argument("--wandb_entity", type=str, default="username-here")
    parser.add_argument("--wandb_proj", type=str, default="GTM")
    parser.add_argument("--wandb_run", type=str, default="Run1")

    args = parser.parse_args()
    run(args)
