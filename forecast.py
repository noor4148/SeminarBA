import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import pytorch_lightning as pl
import torch
from sklearn.metrics import mean_absolute_error
from tqdm import tqdm

from models.FCN import FCN
from models.GTM import GTM
from utils.data_multitrends import ZeroShotDataset


def cal_error_metrics(gt, forecasts):
    mae = mean_absolute_error(gt, forecasts)
    wape = 100 * np.sum(np.sum(np.abs(gt - forecasts), axis=-1)) / np.sum(gt)
    return round(mae, 3), round(wape, 3)


def print_error_metrics(y_test, y_hat, rescaled_y_test, rescaled_y_hat):
    mae, wape = cal_error_metrics(y_test, y_hat)
    rescaled_mae, rescaled_wape = cal_error_metrics(rescaled_y_test, rescaled_y_hat)
    print(mae, wape, rescaled_mae, rescaled_wape)


def run(args):
    print(args)
    device = torch.device(f"cuda:{args.gpu_num}" if torch.cuda.is_available() else "cpu")
    pl.seed_everything(args.seed)

    test_df = pd.read_csv(Path(args.data_folder) / "test.csv", parse_dates=["release_date"])
    test_df = test_df.sort_values("release_date").reset_index(drop=True)
    item_codes = test_df["external_code"].values

    retrieval_pool_df = pd.read_csv(Path(args.data_folder) / "train.csv", parse_dates=["release_date"])
    retrieval_pool_df = retrieval_pool_df.sort_values("release_date").reset_index(drop=True)

    cat_dict = torch.load(Path(args.data_folder) / "category_labels.pt", weights_only=False)
    col_dict = torch.load(Path(args.data_folder) / "color_labels.pt", weights_only=False)
    fab_dict = torch.load(Path(args.data_folder) / "fabric_labels.pt", weights_only=False)

    gtrends = pd.read_csv(Path(args.data_folder) / "gtrends.csv", index_col=[0], parse_dates=True)
    retrieval_cache_dir = Path(args.retrieval_cache_dir) if args.retrieval_cache_dir else Path(args.data_folder) / "retrieval_cache"

    test_loader = ZeroShotDataset(
        test_df,
        Path(args.data_folder) / "images",
        gtrends,
        cat_dict,
        col_dict,
        fab_dict,
        args.trend_len,
        retrieval_bank_df=retrieval_pool_df if args.use_retrieval else None,
        use_retrieval=args.use_retrieval,
        retrieval_top_k=args.retrieval_top_k,
        retrieval_min_similarity=args.retrieval_min_similarity,
        retrieval_observability_weeks=args.retrieval_observability_weeks,
        retrieval_cache_dir=retrieval_cache_dir,
        retrieval_batch_size=args.retrieval_batch_size,
    ).get_loader(batch_size=1, train=False)

    model_savename = f"{args.wandb_run}_model{args.model_output_dim}_eval{args.eval_horizon}"

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
            use_retrieval=args.use_retrieval,
            retrieval_seq_len=args.model_output_dim,
        )

    ckpt = torch.load(args.ckpt_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["state_dict"], strict=True)

    model.to(device)
    model.eval()
    gt, forecasts, attns = [], [], []
    for test_data in tqdm(test_loader, total=len(test_loader), ascii=True):
        with torch.no_grad():
            test_data = [tensor.to(device) for tensor in test_data]
            if len(test_data) == 7:
                item_sales, category, color, textures, temporal_features, gtrends, images = test_data
                retrieval_curve, retrieval_available = None, None
            else:
                item_sales, category, color, textures, temporal_features, gtrends, images, retrieval_curve, retrieval_available = test_data

            y_pred, att = model(
                category,
                color,
                textures,
                temporal_features,
                gtrends,
                images,
                analog_curve=retrieval_curve,
                analog_available=retrieval_available,
            )

            y_pred_np = y_pred.detach().cpu().numpy().reshape(-1)
            y_true_np = item_sales.detach().cpu().numpy().reshape(-1)
            forecasts.append(y_pred_np[: args.eval_horizon])
            gt.append(y_true_np[: args.eval_horizon])
            attns.append(att.detach().cpu().numpy())

    attns = np.stack(attns)
    forecasts = np.array(forecasts)
    gt = np.array(gt)

    scale = float(np.load(Path(args.data_folder) / "normalization_scale.npy"))
    rescale_vals = np.full(args.eval_horizon, scale, dtype=np.float32)

    rescaled_forecasts = forecasts * rescale_vals
    rescaled_gt = gt * rescale_vals
    print_error_metrics(gt, forecasts, rescaled_gt, rescaled_forecasts)

    Path("results").mkdir(parents=True, exist_ok=True)
    torch.save(
        {"results": rescaled_forecasts, "gts": rescaled_gt, "codes": item_codes.tolist()},
        Path("results") / f"{model_savename}.pth",
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Zero-shot sales forecasting")

    parser.add_argument("--data_folder", type=str, default="dataset/")
    parser.add_argument("--ckpt_path", type=str, default="log/path-to-model.ckpt")
    parser.add_argument("--gpu_num", type=int, default=0)
    parser.add_argument("--seed", type=int, default=21)

    parser.add_argument("--model_type", type=str, default="GTM", help="Choose between GTM or FCN")
    parser.add_argument("--use_trends", type=int, default=1)
    parser.add_argument("--use_img", type=int, default=1)
    parser.add_argument("--use_text", type=int, default=1)
    parser.add_argument("--trend_len", type=int, default=52)
    parser.add_argument("--num_trends", type=int, default=3)
    parser.add_argument("--embedding_dim", type=int, default=32)
    parser.add_argument("--hidden_dim", type=int, default=64)
    parser.add_argument("--model_output_dim", type=int, default=12)
    parser.add_argument("--eval_horizon", type=int, default=12)
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

    parser.add_argument("--wandb_run", type=str, default="Run1")

    args = parser.parse_args()

    if args.eval_horizon > args.model_output_dim:
        raise ValueError(
            f"eval_horizon ({args.eval_horizon}) cannot be bigger than "
            f"model_output_dim ({args.model_output_dim})."
        )

    run(args)
