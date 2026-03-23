# Shared similarity backbone for GTM extensions

This folder adds the first reusable layer that both planned extensions need:

- explicit historical retrieval
- competition-aware neighborhood modeling

## What is included

### `similarity_dataset.py`
A dataset that mirrors the existing preprocessing, but keeps:
- `external_code`
- `release_date`
- `split`
- a stable `row_id`

This is important because the original `ZeroShotDataset` drops `external_code`, `season`, `release_date`, and `image_path` while tensorizing the data.

### `similarity_backbone.py`
A wrapper around your current GTM/FCN model that extracts:
- `z_i`: static fused launch embedding
- `g_i`: pooled Google Trends embedding
- `x_i = [z_i || g_i]`: shared backbone embedding
- normalized cosine-search vectors

### `retrieval_index.py`
A reusable similarity index that supports:
- raw cosine top-k search
- retrieval mask: only training products with full horizon observed before target launch
- competition mask: only products active at the same time
- a simple retrieval prior baseline using weighted average of neighbor sales curves

### `build_similarity_backbone.py`
Offline script to build and save the store from your trained checkpoint.

## Recommended first workflow

### 1. Train your GTM baseline as you already do
Keep a fixed 12-week output model.

### 2. Build the shared backbone once
Example:

```bash
python similarity_ext/build_similarity_backbone.py \
  --data_folder dataset/ \
  --ckpt_path log/GTM/your_model.ckpt \
  --output_path artifacts/similarity_backbone.pt \
  --model_type GTM \
  --model_output_dim 12 \
  --precompute_topk 200
```

### 3. Inspect whether retrieved neighbors make sense
Example usage:

```python
from models import load_backbone_store
from models import SimilarityIndex

store = load_backbone_store("artifacts/similarity_backbone.pt")
index = SimilarityIndex(store)

# Top historical analogs for product 100 under a 12-week retrieval rule
neighbors = index.get_retrieval_neighbors(
    item_idx=100,
    horizon_weeks=12,
    top_k=5,
    candidate_pool_size=100,
)
print(neighbors[["external_code", "release_date", "split", "score"]])

# Simple non-trainable retrieval prior baseline
result = index.simple_retrieval_prior(
    item_idx=100,
    horizon_weeks=12,
    top_k=5,
    weighted=True,
    candidate_pool_size=100,
)
print(result["prior_curve"])

# Launch-time competition set
comp = index.get_competition_neighbors(
    item_idx=100,
    top_k=10,
    active_weeks=12,
)
print(comp[["external_code", "release_date", "score"]])
```

## Why this is the right first step

This lets you validate three things before adding trainable retrieval layers:

1. the extracted launch embeddings are sensible
2. the temporal masks prevent leakage
3. the same backbone can serve both extensions

## Next step after this

Once this backbone is stable, the next file to build should be a `retrieval_module.py` that:
- takes `x_i` as query input
- retrieves top-k admissible analogs
- projects neighbor sales curves
- fuses the retrieval prior back into GTM

Only after that would I add the competition-aware interaction MLP.
