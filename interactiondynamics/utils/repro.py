import random
import os
import numpy as np
import torch

# for reproducability! = try to make runs with the same seed behave the same
# needed for ML b/c randomness affects initailization/sampling/GPU kernels
def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    #added to see if i can fix seed reproducability issue
    os.environ["PYTHONHASHSEED"] = str(seed)
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True)
    
