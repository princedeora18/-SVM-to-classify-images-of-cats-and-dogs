
import random
import numpy as np

# set seed
random.seed(42)
np.random.seed(42)

# config settings
IMAGE_SIZE = (64, 64)
NUM_SAMPLES_PER_CLASS = 1000
USE_HOG_FEATURES = True
USE_PCA = False
PCA_COMPONENTS = 200
