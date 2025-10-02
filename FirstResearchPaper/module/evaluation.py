import numpy as np
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

def evaluate_performance(ground_truth, predictions, class_names):
    """
    Calculates and prints a classification report and confusion matrix.

    Args:
        ground_truth (np.ndarray): Array of true labels.
        predictions (np.ndarray): Array of predicted labels.
        class_names (list): List of class names for the confusion matrix plot.
    """
    print("\n--- Classification Report ---")
    report = classification_report(ground_truth, predictions, target_names=class_names)
    print(report)

    print("\n--- Confusion Matrix ---")
    cm = confusion_matrix(ground_truth, predictions)
    print(cm)

    # Plotting the confusion matrix
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=class_names, yticklabels=class_names)
    plt.title("Confusion Matrix")
    plt.ylabel("Actual Label")
    plt.xlabel("Predicted Label")
    plt.show()

    accuracy = np.sum(ground_truth == predictions) / len(ground_truth) * 100
    print(f"\nOverall Accuracy: {accuracy:.2f}%")
    
    return report, cm, accuracy
