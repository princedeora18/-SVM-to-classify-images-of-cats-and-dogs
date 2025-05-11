
import matplotlib.pyplot as plt

# show image results
def visualize_results(images, labels, predictions, indices):
    plt.figure(figsize=(12, 8))
    for i, idx in enumerate(indices[:9]):
        plt.subplot(3, 3, i+1)
        plt.imshow(images[idx], cmap='gray')
        true_label = "Cat" if labels[idx] == 0 else "Dog"
        pred_label = "Cat" if predictions[idx] == 0 else "Dog"
        plt.title(f"True: {true_label}, Pred: {pred_label}",
                  color='green' if true_label == pred_label else 'red')
        plt.axis('off')
    plt.tight_layout()
    plt.savefig("svm_predictions.png")
    plt.close()
