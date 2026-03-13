"""Merge Z-Image Turbo sharded safetensors into single files for ComfyUI."""
import json
import os
from pathlib import Path
from safetensors.torch import load_file, save_file

SNAP = Path.home() / ".cache/huggingface/hub/models--Tongyi-MAI--Z-Image-Turbo/snapshots/f332072aa78be7aecdf3ee76d5c247082da564a6"
OUT = Path.home() / "ComfyUI/models"

def merge_shards(index_path, output_path, label):
    print(f"\n=== Merging {label} ===")
    with open(index_path) as f:
        index = json.load(f)
    weight_map = index["weight_map"]
    shard_dir = Path(index_path).parent
    shards = sorted(set(weight_map.values()))

    merged = {}
    for shard in shards:
        print(f"  Loading {shard} ...")
        tensors = load_file(str(shard_dir / shard))
        merged.update(tensors)
        print(f"  -> {len(tensors)} tensors, total so far: {len(merged)}")

    print(f"  Saving to {output_path} ...")
    os.makedirs(Path(output_path).parent, exist_ok=True)
    save_file(merged, str(output_path))
    size = Path(output_path).stat().st_size / 1e9
    print(f"  Done! {size:.1f} GB")

# 1. Merge transformer
merge_shards(
    SNAP / "transformer/diffusion_pytorch_model.safetensors.index.json",
    OUT / "diffusion_models/z_image_turbo.safetensors",
    "Transformer"
)

# 2. Merge text encoder (Qwen3)
merge_shards(
    SNAP / "text_encoder/model.safetensors.index.json",
    OUT / "text_encoders/qwen3_z_image.safetensors",
    "Text Encoder (Qwen3)"
)

print("\n=== All done! ===")
print("Models ready in ComfyUI/models/")
