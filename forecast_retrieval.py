import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import pytorch_lightning as pl
import torch
from sklearn.metrics import mean_absolute_error
from tqdm import tqdm

from models.GTM import GTM
from models.GTM_retrieval import GTMRetrieval
from utils.data_retrieval import RetrievalTrendDataset
from utils.retrieval_metadata import load_split_dataframes


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
    device = torch.device(f'cuda:{args.gpu_num}' if torch.cuda.is_available() else 'cpu')
    pl.seed_everything(args.seed)

    _, _, test_df = load_split_dataframes(data_folder=args.data_folder, val_fraction=args.val_fraction)
    item_codes = test_df['external_code'].values
    gtrends = pd.read_csv(Path(args.data_folder) / 'gtrends.csv', index_col=[0], parse_dates=True)

    test_loader = RetrievalTrendDataset(
        test_df,
        gtrends=gtrends,
        trend_len=args.trend_len,
        sales_horizon=args.model_output_dim,
    ).get_loader(batch_size=1, shuffle=False, num_workers=args.num_workers)

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

    model = GTMRetrieval(
        baseline_gtm=baseline_gtm,
        retrieval_store_path=args.embedding_store_path,
        retrieval_observation_horizon=args.retrieval_observation_horizon,
        retrieval_k=args.retrieval_k,
        retrieval_projector_hidden_dim=args.retrieval_projector_hidden_dim,
        augment_hidden_dim=args.augment_hidden_dim,
        normalization_scale_value=args.normalization_scale_value,
    )

    ckpt = torch.load(args.ckpt_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt['state_dict'], strict=True)
    model.to(device)
    model.eval()

    gt = []
    forecasts = []
    retrieval_logs = []
    attns = []
    for test_data in tqdm(test_loader, total=len(test_loader), ascii=True):
        with torch.no_grad():
            item_sales, multitrends, batch_codes, split_names = test_data
            multitrends = multitrends.to(device)
            item_sales = item_sales.to(device)
            batch_codes = list(batch_codes)
            split_name = str(split_names[0])
            y_pred, att, retrieval_output = model(batch_codes, multitrends, target_split=split_name, return_retrieval_info=True)

            y_pred_np = y_pred.detach().cpu().numpy().reshape(-1)
            y_true_np = item_sales.detach().cpu().numpy().reshape(-1)
            forecasts.append(y_pred_np[: args.eval_horizon])
            gt.append(y_true_np[: args.eval_horizon])
            attns.append(att.detach().cpu().numpy())

            detail = retrieval_output['details'][0]
            detail['forecast'] = y_pred_np[: args.eval_horizon].tolist()
            detail['ground_truth'] = y_true_np[: args.eval_horizon].tolist()
            retrieval_logs.append(detail)

    attns = np.stack(attns)
    forecasts = np.array(forecasts)
    gt = np.array(gt)

    # Rescale the values in such a way that it won't end up with a 0-dimentional vector
    scale = float(np.load(Path(args.data_folder) / 'normalization_scale.npy'))
    rescale_vals = np.full(args.eval_horizon, scale, dtype=np.float32)

    #rescale_vals = np.load(Path(args.data_folder) / 'normalization_scale.npy')[: args.eval_horizon]
    rescaled_forecasts = forecasts * rescale_vals
    rescaled_gt = gt * rescale_vals
    print_error_metrics(gt, forecasts, rescaled_gt, rescaled_forecasts)

    Path('results').mkdir(parents=True, exist_ok=True)
    model_savename = f'{args.run_name}_retrieval_model{args.model_output_dim}_eval{args.eval_horizon}'
    torch.save(
        {
            'results': rescaled_forecasts,
            'gts': rescaled_gt,
            'codes': item_codes.tolist(),
            'attns': attns,
            'retrieval_logs': retrieval_logs,
        },
        Path('results') / f'{model_savename}.pth',
    )


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Forecast with retrieval-augmented GTM.')
    parser.add_argument('--data_folder', type=str, default='dataset/')
    parser.add_argument('--embedding_store_path', type=str, required=True)
    parser.add_argument('--ckpt_path', type=str, required=True)
    parser.add_argument('--run_name', type=str, default='Run1')
    parser.add_argument('--gpu_num', type=int, default=0)
    parser.add_argument('--seed', type=int, default=21)
    parser.add_argument('--num_workers', type=int, default=0)
    parser.add_argument('--val_fraction', type=float, default=0.15)

    parser.add_argument('--use_img', type=int, default=1)
    parser.add_argument('--use_text', type=int, default=1)
    parser.add_argument('--trend_len', type=int, default=52)
    parser.add_argument('--num_trends', type=int, default=3)
    parser.add_argument('--embedding_dim', type=int, default=32)
    parser.add_argument('--hidden_dim', type=int, default=64)
    parser.add_argument('--model_output_dim', type=int, default=12)
    parser.add_argument('--eval_horizon', type=int, default=12)
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

    if args.eval_horizon > args.model_output_dim:
        raise ValueError(
            f'eval_horizon ({args.eval_horizon}) cannot be bigger than model_output_dim ({args.model_output_dim}).'
        )
    run(args)