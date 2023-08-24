import os
import shutil
from src.utils import format_and_copy_json

# Change these parameters
NITERS      = 100
BASE_CONFIG = "configs_loop_sunlamp_10_epoch"
BASE_FILE   = "loop_sunlamp_niter_0000.json"

for i in range(1, NITERS):

    print("Training Loop Sunlamp - NITER ", i)

    # Create a new config file from the base path
    target_config_name = BASE_FILE.split("niter")[0] + "niter_" + str(i).zfill(4) + ".json"

    target_config_path = os.path.join(BASE_CONFIG, target_config_name)
    base_config_path   = os.path.join(BASE_CONFIG, BASE_FILE)
    shutil.copyfile(base_config_path, target_config_path)

    # The idea is the following
    # We have a good trained model at t0
    # We generate the pseudolabels with t0
    # We clean the pseudolabels from t0
    # We train from pre-trained, with the pseudolabels

    # Generate pseudo-labels with the iter - 1

    # Take the initial model and make pseudolabels with it (t-1)

    previous_file   = BASE_FILE.split("niter")[0] + "niter_" + str(i-1).zfill(4) + ".json"
    previous_config = os.path.join(BASE_CONFIG, previous_file)
    os.system("python generate_pseudo_labels.py --cfg " +  previous_config)    
    format_and_copy_json(previous_config)

    os.system("python create_maps.py --cfg " + target_config_path)
    ## Train the new model with the labels
    os.system("python main.py --cfg " + target_config_path)

