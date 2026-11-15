"""Resume/complete the Qwen2.5-1.5B-Instruct snapshot download into bench/hf_cache.
HF hub resumes .incomplete partial blobs automatically."""
import os

from huggingface_hub import snapshot_download

BASE = os.path.dirname(os.path.abspath(__file__))
path = snapshot_download(
    "Qwen/Qwen2.5-1.5B-Instruct",
    cache_dir=os.path.join(BASE, "hf_cache"),
)
print("DOWNLOAD_COMPLETE", path, flush=True)
