import pickle
from sklearn.metrics import roc_auc_score, roc_curve
import matplotlib.pyplot as plt

# Load probabilities and true labels
model_files = ["alexnet_labels_data.pkl", "resnet_labels_data.pkl", "vgg_labels_data.pkl", "squeeze_labels_data.pkl"]
model_names = ["alexnet", "resnet", "vgg", "squeeze"]
colors = ["b", "orange", "g", "r"]

plt.figure()

for model_file, model_name, color in zip(model_files, model_names, colors):
    with open(model_file, "rb") as f:
        data = pickle.load(f)
        true_labels = data["true_labels"]
        predicted_probs = data["pred_labels"]

        # Calculate AUC and ROC Curve
        auc_score = roc_auc_score(true_labels, predicted_probs)
        fpr, tpr, _ = roc_curve(true_labels, predicted_probs)

        # Plot ROC curve
        plt.plot(fpr, tpr, color=color, label=f"{model_name} (AUC = {auc_score:.2f})")

plt.plot([0, 1], [0, 1], "k--", label="Random Guessing")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve Comparison")
plt.legend()
plt.show()