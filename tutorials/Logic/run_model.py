from lfads_torch.run_model import run_model
from datetime import datetime
from pathlib import Path


PROJECT_STR = "lfads-torch-logic"
DATASET_STR = "logic_sr"
RUN_TAG = datetime.now().strftime("%y%m%d")
RUN_DIR = Path("/path/to/save/run/") / PROJECT_STR / DATASET_STR / RUN_TAG

mandatory_overrides = {
    "datamodule": DATASET_STR,
    "model": DATASET_STR,

}

best_ckpt_dir = "/user_data/jpulidoa/lfads-torch/runs/lfads-torch-logic/logic_sr/260318_logicSingle/lightning_checkpoints/"

run_model(
    overrides=mandatory_overrides,
    checkpoint_dir=best_ckpt_dir,
    config_path="../configs/single.yaml",
    do_train=False,
)

