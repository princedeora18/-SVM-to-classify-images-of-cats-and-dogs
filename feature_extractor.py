
from skimage.feature import hog
import numpy as np
from tqdm import tqdm

# extract features (either HOG or raw pixels)
def extract_features(images, use_hog=True):
    features = []
    if use_hog:
        print("Extracting HOG features...")
        for img in tqdm(images):
            feat = hog(img, orientations=9, pixels_per_cell=(8, 8),
                       cells_per_block=(2, 2), visualize=False)
            features.append(feat)
    else:
        print("Using raw pixels...")
        for img in tqdm(images):
            features.append(img.flatten())
    return np.array(features)
