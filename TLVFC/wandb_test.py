import os
os.environ["WANDB_MODE"] = "offline"

import wandb

wandb.init(project="test-wandb", name="test_run")

for i in range(5):
    wandb.log({"test_value": i})
