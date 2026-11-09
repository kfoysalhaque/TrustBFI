# fetch_dataset.py
# Pull the 135 V-matrix files (4.6 GB) into Dataset/vmatrices/.
#
#   python fetch_dataset.py
#
# The dataset is public: https://huggingface.co/datasets/foysalhaque/BeamID

import os
from huggingface_hub import snapshot_download

REPO = "foysalhaque/BeamID"
DEST = os.path.join(os.path.dirname(os.path.abspath(__file__)), "Dataset")

if __name__ == "__main__":
    p = snapshot_download(
        repo_id=REPO,
        repo_type="dataset",
        allow_patterns=["vmatrices/*.npy"],
        local_dir=DEST,
        max_workers=8,
    )
    print("[fetch] dataset at:", os.path.join(p, "vmatrices"))
