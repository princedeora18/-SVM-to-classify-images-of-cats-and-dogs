
import os
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA

from config import *
from data_loader import load_and_preprocess_data
from feature_extractor import extract_features
from svm_model import train_and_evaluate_svm
from visualize import visualize_results
from predict_unlabeled import predict_unlabeled

def main():
    data_dir = "./dogs-vs-cats"

    train_dir = os.path.join(data_dir, "train")
    if not os.path.exists(train_dir):
        print("Dataset not found.")
        return

    images, labels = load_and_preprocess_data(data_dir, NUM_SAMPLES_PER_CLASS)
    features = extract_features(images, use_hog=USE_HOG_FEATURES)

    pca = None
    if USE_PCA:
        print("Applying PCA...")
        pca = PCA(n_components=PCA_COMPONENTS, random_state=42)
        features = pca.fit_transform(features)
        print("PCA applied.")

    X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.2, random_state=42)

    model, acc, report = train_and_evaluate_svm(X_train, X_test, y_train, y_test, "rbf")

    indices = np.random.choice(len(X_test), size=9, replace=False)
    test_indices = np.arange(len(features))
    _, test_full_indices = train_test_split(test_indices, test_size=0.2, random_state=42)
    sample_idx = test_full_indices[indices]
    visualize_results(images, labels, model.predict(features), sample_idx)

    print("RBF evaluation done. Trying linear kernel...")
    train_and_evaluate_svm(X_train, X_test, y_train, y_test, "linear")

    predict_unlabeled(model, pca)

if __name__ == "__main__":
    main()
