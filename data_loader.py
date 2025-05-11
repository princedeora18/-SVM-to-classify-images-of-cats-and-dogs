
import os
import cv2
import numpy as np
import random
from tqdm import tqdm

from config import IMAGE_SIZE

# function to load & preprocess images
def load_and_preprocess_data(data_dir, num_samples_per_class=1000):
    
    print("Loading and preprocessing data...")

    images, labels = [], []
    train_dir = os.path.join(data_dir, "train")

    # load cat images
    cat_files = [f for f in os.listdir(train_dir) if f.startswith('cat.')]
    if len(cat_files) > num_samples_per_class:
        cat_files = random.sample(cat_files, num_samples_per_class)
        
    for filename in tqdm(cat_files, desc="Loading cat images"):
        try:
            path = os.path.join(train_dir, filename)
            img = cv2.imread(path)
            if img is None:
                continue
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            img = cv2.resize(img, IMAGE_SIZE)
            images.append(img)
            labels.append(0)
        except:
            continue

    # load dog images
    dog_files = [f for f in os.listdir(train_dir) if f.startswith('dog.')]
    if len(dog_files) > num_samples_per_class:
        dog_files = random.sample(dog_files, num_samples_per_class)
        
    for filename in tqdm(dog_files, desc="Loading dog images"):
        try:
            path = os.path.join(train_dir, filename)
            img = cv2.imread(path)
            if img is None:
                continue
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            img = cv2.resize(img, IMAGE_SIZE)
            images.append(img)
            labels.append(1)
        except:
            continue

    return np.array(images), np.array(labels)
