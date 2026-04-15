import shutil
from datetime import datetime
from pathlib import Path
import os

from ray import tune
from ray.tune import CLIReporter
from ray.tune.schedulers import FIFOScheduler
from ray.tune.suggest.basic_variant import BasicVariantGenerator

from lfads_torch.run_model import run_model

# ---------- OPTIONS -----------
PROJECT_STR = "lfads-torch-logic-251209_preparatory"
DATASET_STR = "logic_sr_251209_preparatory" #  
MODEL_STR = "logic_sr_251209_preparatory" # 
RUN_TAG = datetime.now().strftime("%y%m%d%H%M") + "_logicMulti"
RUN_DIR = Path("runs")  / PROJECT_STR / DATASET_STR / RUN_TAG
# ------------------------------

# Set the mandatory config overrides to select datamodule and model
mandatory_overrides = {
    "datamodule": DATASET_STR,
    "model": MODEL_STR,
    "logger.wandb_logger.project": PROJECT_STR,
    "logger.wandb_logger.tags.1": DATASET_STR,
    "logger.wandb_logger.tags.2": RUN_TAG,
}
RUN_DIR.mkdir(parents=True)
# Copy this script into the run directory
#os.chdir(RUN_DIR)
shutil.copyfile(__file__, RUN_DIR / Path(__file__).name)
# Run the hyperparameter search
tune.run(
    tune.with_parameters(
        run_model,
        config_path="../configs/multi.yaml",
    ),
    metric="valid/recon_smth",
    mode="min",
    name=RUN_DIR.name,
    config={
        **mandatory_overrides,
        "model.dropout_rate": tune.uniform(0.2, 0.5),
        "model.kl_co_scale": tune.loguniform(1e-6, 1e-4),
        "model.kl_ic_scale": tune.loguniform(1e-6, 1e-3),
        "model.l2_gen_scale": tune.loguniform(1e-4, 1e0),
        "model.l2_con_scale": tune.loguniform(1e-4, 1e0),
    },
    resources_per_trial=dict(cpu=1, gpu=0.25),
    num_samples=60,
    local_dir=RUN_DIR.parent,
    search_alg=BasicVariantGenerator(random_state=0),
    scheduler=FIFOScheduler(),
    verbose=1,
    progress_reporter=CLIReporter(
        metric_columns=["valid/recon_smth", "cur_epoch"],
        sort_by_metric=True,
    ),
    trial_dirname_creator=lambda trial: str(trial),
)
