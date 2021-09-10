
import os
import shutil
from optirv import env_config, gen_dataset_metafile, upload_dataset
env_config("config.json")

# local destination directory
dst_dir = os.path.join(os.environ.get("DATA_DIR"), "optirv-configs")

# check that metadata file exists for Kaggle Datasets API else create
if os.path.exists(os.path.join(dst_dir, "dataset-metadata.json")):
    pass
else:
    gen_dataset_metafile(dst_dir, user="slashie",
                         title="OptiRV Config Files",
                         subtitle="Config files to use with "+
                         "the OptiRV package")

# upload to Kaggle datasets
upload_dataset(dst_dir,
               new_version=True,
               version_notes="Upload new final proc (class) config")
