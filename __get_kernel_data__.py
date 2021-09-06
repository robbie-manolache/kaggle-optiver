
import os
from optirv import env_config, kernel_output_download

# set config and data directory
env_config("config.json")
user = "slashie"
kernel = "orvp-feat-eng-001"
local_path = os.path.join(os.environ.get("DATA_DIR"), 
                          "kaggle-kernel-download")

# run download
kernel_output_download(user, kernel, local_path)
