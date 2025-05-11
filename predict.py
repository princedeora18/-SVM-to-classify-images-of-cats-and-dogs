
import os
import cv2
import matplotlib.pyplot as plt
from config import IMAGE_SIZE, USE_HOG_FEATURES, USE_PCA
from skimage.feature import hog

# predict unlabelled test images
def predict_unlabeled(model, pca=None):
    test1_dir = './dogs-vs-cats/test1/test1/'
    if not os.path.exists(test1_dir):
        return

    files = os.listdir(test1_dir)
    files = files[:5]
    plt.figure(figsize=(15, 3))

    for i, f in enumerate(files):
        path = os.path.join(test1_dir, f)
        img = cv2.imread(path)
        if img is None:
            continue
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        resized = cv2.resize(gray, IMAGE_SIZE)
        features = hog(resized, orientations=9, pixels_per_cell=(8, 8),
                       cells_per_block=(2, 2), visualize=False) if USE_HOG_FEATURES else resized.flatten()
        if USE_PCA and pca:
            features = pca.transform([features])[0]
        pred = model.predict([features])[0]
        prob = model.predict_proba([features])[0]
        conf = max(prob) * 100
        label = "Cat" if pred == 0 else "Dog"
        plt.subplot(1, 5, i+1)
        plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        plt.title(f"{label}\n{conf:.1f}%")
        plt.axis('off')

    plt.tight_layout()
    plt.savefig("test_predictions.png")
    plt.close()
