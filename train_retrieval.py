import argparse
from datetime import datetime
from pathlib import Path

import pytorch_lightning as pl
from pytorch_lightning import loggers as pl_loggers
import pandas as pd
import torch

from models.GTM import GTM
from models.GTM_retrieval import GTMRetrieval
from utils.data_retrieval import RetrievalTrendDataset
from utils.retrieval_metadata import load_split_dataframes


def run(args):
    print(args)
    pl.seed_everything(args.seed)

    subtrain_df, val_df, _ = load_split_dataframes(data_folder=args.data_folder, val_fraction=args.val_fraction)
    gtrends = pd.read_csv(Path(args.data_folder) / 'gtrends.csv', index_col=[0], parse_dates=True)

    train_loader = RetrievalTrendDataset(
        subtrain_df,
        gtrends=gtrends,
        trend_len=args.trend_len,
        sales_horizon=args.model_output_dim,
    ).get_loader(batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)

    val_loader = RetrievalTrendDataset(
        val_df,
        gtrends=gtrends,
        trend_len=args.trend_len,
        sales_horizon=args.model_output_dim,
    ).get_loader(batch_size=args.val_batch_size, shuffle=False, num_workers=args.num_workers)

    cat_dict = torch.load(Path(args.data_folder) / 'category_labels.pt', weights_only=False)
    col_dict = torch.load(Path(args.data_folder) / 'color_labels.pt', weights_only=False)
    fab_dict = torch.load(Path(args.data_folder) / 'fabric_labels.pt', weights_only=False)

    baseline_gtm = GTM(
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
    baseline_ckpt = torch.load(args.baseline_ckpt_path, map_location='cpu', weights_only=False)
    baseline_gtm.load_state_dict(baseline_ckpt['state_dict'], strict=True)

    model = GTMRetrieval(
        baseline_gtm=baseline_gtm,
        retrieval_store_path=args.embedding_store_path,
        retrieval_observation_horizon=args.retrieval_observation_horizon,
        retrieval_k=args.retrieval_k,
        retrieval_projector_hidden_dim=args.retrieval_projector_hidden_dim,
        augment_hidden_dim=args.augment_hidden_dim,
        normalization_scale_value=args.normalization_scale_value,
    )

    dt_string = datetime.now().strftime('%d-%m-%Y-%H-%M-%S')
    model_savename = f'GTM_Retrieval_{args.run_name}'
    checkpoint_callback = pl.callbacks.ModelCheckpoint(
        dirpath=Path(args.log_dir) / 'GTM_Retrieval',
        filename=model_savename + '---{epoch}---' + dt_string,
        monitor='val_mae',
        mode='min',
        save_top_k=1,
    )
    tb_logger = pl_loggers.TensorBoardLogger(save_dir=args.log_dir, name=model_savename)

    trainer_kwargs = dict(
        max_epochs=args.epochs,
        check_val_every_n_epoch=args.check_val_every_n_epoch,
        logger=tb_logger,
        callbacks=[checkpoint_callback],
    )
    if torch.cuda.is_available():
        trainer_kwargs['gpus'] = [args.gpu_num]
    trainer = pl.Trainer(**trainer_kwargs)
    trainer.fit(model, train_dataloaders=train_loader, val_dataloaders=val_loader)
    print(checkpoint_callback.best_model_path)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train explicit-retrieval GTM.')
    parser.add_argument('--data_folder', type=str, default='dataset/')
    parser.add_argument('--baseline_ckpt_path', type=str, required=True)
    parser.add_argument('--embedding_store_path', type=str, required=True)
    parser.add_argument('--log_dir', type=str, default='log')
    parser.add_argument('--run_name', type=str, default='Run1')
    parser.add_argument('--seed', type=int, default=21)
    parser.add_argument('--epochs', type=int, default=200)
    parser.add_argument('--gpu_num', type=int, default=0)
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--val_batch_size', type=int, default=32)
    parser.add_argument('--num_workers', type=int, default=0)
    parser.add_argument('--check_val_every_n_epoch', type=int, default=5)
    parser.add_argument('--val_fraction', type=float, default=0.15)

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

    parser.add_argument('--retrieval_observation_horizon', type=int, default=12)
    parser.add_argument('--retrieval_k', type=int, default=5)
    parser.add_argument('--retrieval_projector_hidden_dim', type=int, default=64)
    parser.add_argument('--augment_hidden_dim', type=int, default=64)
    parser.add_argument('--normalization_scale_value', type=float, default=1065.0)
    args = parser.parse_args()
    run(args)