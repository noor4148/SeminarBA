import copy

import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers.optimization import Adafactor

from utils.embedding_store import RetrievalStore
from utils.retriever import TemporalNearestNeighborRetriever


class GTMRetrieval(pl.LightningModule):
    def __init__(
        self,
        baseline_gtm,
        retrieval_store_path,
        retrieval_observation_horizon=12,
        retrieval_k=5,
        retrieval_projector_hidden_dim=128,
        augment_hidden_dim=128,
        normalization_scale_value=1065.0,
    ):
        super().__init__()
        self.save_hyperparameters(ignore=['baseline_gtm'])

        self.hidden_dim = baseline_gtm.hidden_dim
        self.output_len = baseline_gtm.output_len
        self.autoregressive = baseline_gtm.autoregressive
        self.normalization_scale_value = normalization_scale_value

        self.gtrend_encoder = copy.deepcopy(baseline_gtm.gtrend_encoder)
        self.decoder = copy.deepcopy(baseline_gtm.decoder)
        self.decoder_fc = copy.deepcopy(baseline_gtm.decoder_fc)
        self.pos_encoder = copy.deepcopy(getattr(baseline_gtm, 'pos_encoder', None))

        self.trajectory_projector = nn.Sequential(
            nn.Linear(retrieval_observation_horizon, retrieval_projector_hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(retrieval_projector_hidden_dim, self.hidden_dim),
        )
        self.augment_mlp = nn.Sequential(
            nn.Linear(self.hidden_dim * 2 + 1, augment_hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(augment_hidden_dim, self.hidden_dim),
        )

        self.retrieval_store = RetrievalStore.load(retrieval_store_path)
        self.retriever = TemporalNearestNeighborRetriever(
            store=self.retrieval_store,
            retrieval_k=retrieval_k,
            retrieval_observation_horizon=retrieval_observation_horizon,
        )

    def _generate_square_subsequent_mask(self, size, device):
        mask = (torch.triu(torch.ones(size, size, device=device)) == 1).transpose(0, 1)
        return mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))

    def _ensure_retriever_device(self, device):
        if self.retriever.device != torch.device(device):
            self.retriever.to(device)

    def forward(self, item_codes, gtrends, target_split, return_retrieval_info=False):
        self._ensure_retriever_device(gtrends.device)
        retrieval_output = self.retriever.retrieve_batch(item_codes, target_split=target_split, device=gtrends.device)

        neighbor_sales = retrieval_output['neighbor_sales']
        weights = retrieval_output['weights']
        availability = retrieval_output['availability']
        target_embeddings = retrieval_output['target_embeddings']

        batch_size, num_neighbors, sales_dim = neighbor_sales.shape
        projected_neighbors = self.trajectory_projector(neighbor_sales.view(batch_size * num_neighbors, sales_dim))
        projected_neighbors = projected_neighbors.view(batch_size, num_neighbors, self.hidden_dim)
        retrieval_vector = (projected_neighbors * weights.unsqueeze(-1)).sum(dim=1)

        augmented_static = self.augment_mlp(torch.cat([target_embeddings, retrieval_vector, availability], dim=-1))
        gtrend_encoding = self.gtrend_encoder(gtrends)

        if self.autoregressive == 1:
            tgt = torch.zeros(
                self.output_len,
                gtrend_encoding.shape[1],
                gtrend_encoding.shape[-1],
                device=gtrend_encoding.device,
            )
            tgt[0] = augmented_static
            tgt = self.pos_encoder(tgt)
            tgt_mask = self._generate_square_subsequent_mask(self.output_len, gtrend_encoding.device)
            decoder_out, attn_weights = self.decoder(tgt, gtrend_encoding, tgt_mask)
            forecast = self.decoder_fc(decoder_out)
        else:
            tgt = augmented_static.unsqueeze(0)
            decoder_out, attn_weights = self.decoder(tgt, gtrend_encoding)
            forecast = self.decoder_fc(decoder_out)

        forecast = forecast.view(-1, self.output_len)
        if return_retrieval_info:
            return forecast, attn_weights, retrieval_output
        return forecast, attn_weights

    def configure_optimizers(self):
        #optimizer = Adafactor(self.parameters(), scale_parameter=True, relative_step=True, warmup_init=True, lr=None)
        optimizer = torch.optim.AdamW(self.parameters(), lr=1e-3)
        return [optimizer]

    def training_step(self, train_batch, batch_idx):
        item_sales, gtrends, item_codes, split_names = train_batch
        target_split = str(split_names[0])
        forecasted_sales, _ = self.forward(list(item_codes), gtrends, target_split=target_split)
        loss = F.mse_loss(item_sales, forecasted_sales)
        self.log('train_loss', loss)
        return loss

    def validation_step(self, val_batch, batch_idx):
        item_sales, gtrends, item_codes, split_names = val_batch
        target_split = str(split_names[0])
        forecasted_sales, _ = self.forward(list(item_codes), gtrends, target_split=target_split)
        return item_sales, forecasted_sales

    def validation_epoch_end(self, val_step_outputs):
        item_sales = torch.cat([x[0] for x in val_step_outputs], dim=0)
        forecasted_sales = torch.cat([x[1] for x in val_step_outputs], dim=0)
        rescaled_item_sales = item_sales * self.normalization_scale_value
        rescaled_forecasted_sales = forecasted_sales * self.normalization_scale_value
        loss = F.mse_loss(item_sales, forecasted_sales)
        mae = F.l1_loss(rescaled_item_sales, rescaled_forecasted_sales)
        self.log('val_mae', mae)
        self.log('val_loss', loss)
        print('Validation MAE:', mae.detach().cpu().numpy(), 'LR:', self.optimizers().param_groups[0]['lr'])
