
import os
from optirv import env_config, kernel_output_download

# set config and data directory
env_config("config.json")
user = "slashie"
kernels = ["orvp-lgbm-reg-0%d"%i for i in [1,2,3,4,5]]
local_path = os.path.join(os.environ.get("DATA_DIR"), 
                          "kaggle-kernel-download")

# run download
for kernel in kernels:
    kernel_output_download(user, kernel, local_path)
