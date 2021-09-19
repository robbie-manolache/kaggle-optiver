
import os
import shutil
from optirv import env_config, gen_dataset_metafile, upload_dataset
env_config("config.json")

# local destination directory
dst_dir = os.path.join(os.environ.get("DATA_DIR"), "optirv-py-pkg")

# check that metadata file exists for Kaggle Datasets API else create
if os.path.exists(os.path.join(dst_dir, "dataset-metadata.json")):
    pass
else:
    gen_dataset_metafile(dst_dir, user="slashie",
                         title="OptiRV Python Package",
                         subtitle="Python package for Optiver "+
                         "Realized Volatility Prediction competition")

# get package file name
file_name = "optirv-0.17-py3-none-any.whl"

# copy source file to destination    
shutil.copy2(os.path.join("dist", file_name), 
             os.path.join(dst_dir, file_name))

# upload to Kaggle datasets
upload_dataset(dst_dir,
               new_version=True,
               version_notes="Upload 0.17 version")
