import numpy as np
from load_images import load_images
from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
from sklearn.metrics import recall_score
from sklearn.metrics import precision_score, recall_score, f1_score
import pickle


directory = 'sickle-cell-img'
df = load_images(directory)
#print(df.head())

X = np.array(df['image'].tolist())
y = np.array(df['label'])

X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

#vgg19 implementation
model_squeeze = torch.hub.load('pytorch/vision:v0.10.0', 'squeezenet1_1', pretrained=True)
model_squeeze.classifier[1] = nn.Conv2d(512, 2, kernel_size=(1,1), stride=(1,1))
model_squeeze.num_classes = 2
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model_squeeze = model_squeeze.to(device)

def get_class_weights(labels):
    class_sample_count = [len(labels[labels == 0]), len(labels[labels == 1])] 
    total_samples = len(labels)
    weights = [total_samples / count for count in class_sample_count]
    return torch.tensor(weights, dtype=torch.float)

class_weights = torch.tensor([3.0, 1.0], dtype=torch.float)
criterion = nn.CrossEntropyLoss(weight=class_weights.to(device))
optimizer = optim.Adam(model_squeeze.parameters(), lr=0.0001)

# Transforms for training, validation, and test datasets
train_transforms = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((224, 224)),  # Resize to ResNet's input size
    transforms.RandomHorizontalFlip(),  # Augmentation
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # ImageNet normalization
])

val_test_transforms = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((224, 224)),  # Resize to ResNet's input size
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

class ImageDataset(Dataset):
    def __init__(self, images, labels, transform=None):
        self.images = images
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = self.images[idx].astype(np.uint8)
        label = self.labels[idx]

        if self.transform:
            image = self.transform(image)

        return image, label

# Create dataset instances for train, validation, and test sets
train_dataset = ImageDataset(X_train, y_train, transform=train_transforms)
val_dataset = ImageDataset(X_val, y_val, transform=val_test_transforms)
test_dataset = ImageDataset(X_test, y_test, transform=val_test_transforms)

# Create DataLoaders for batching
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

# Function to train the model with class imbalance handling and recall focus
def train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs=10):
    model.train()
    for epoch in range(num_epochs):
        running_loss = 0.0
        total_train = 0

        true_pos_train = 0  # Correctly predicted positives
        false_neg_train = 0  # Positives predicted as negatives
        true_neg_train = 0  # Correctly predicted negatives
        false_pos_train = 0  # Negatives predicted as positives

        # Training loop
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            total_train += labels.size(0)

            true_pos_train += ((predicted == 1) & (labels == 1)).sum().item()
            false_neg_train += ((predicted == 0) & (labels == 1)).sum().item()
            true_neg_train += ((predicted == 0) & (labels == 0)).sum().item()
            false_pos_train += ((predicted == 1) & (labels == 0)).sum().item()

        recall_pos_train = true_pos_train / (true_pos_train + false_neg_train) if (true_pos_train + false_neg_train) > 0 else 0
        recall_neg_train = true_neg_train / (true_neg_train + false_pos_train) if (true_neg_train + false_pos_train) > 0 else 0

        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {running_loss/len(train_loader):.4f}, '
              f'Training Recall (Pos): {recall_pos_train:.4f}, Training Recall (Neg): {recall_neg_train:.4f}')

        # Validate the model
        validate_model(model, val_loader, criterion)

# Function to validate the model and calculate recall
def validate_model(model, val_loader, criterion):
    model.eval()
    val_loss = 0.0
    total_val = 0

    true_pos_val = 0  # Correctly predicted positives
    false_neg_val = 0  # Positives predicted as negatives
    true_neg_val = 0  # Correctly predicted negatives
    false_pos_val = 0  # Negatives predicted as positives

    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            val_loss += loss.item()
            
            _, predicted = torch.max(outputs, 1)
            total_val += labels.size(0)

            true_pos_val += ((predicted == 1) & (labels == 1)).sum().item()
            false_neg_val += ((predicted == 0) & (labels == 1)).sum().item()
            true_neg_val += ((predicted == 0) & (labels == 0)).sum().item()
            false_pos_val += ((predicted == 1) & (labels == 0)).sum().item()

    recall_pos_val = true_pos_val / (true_pos_val + false_neg_val) if (true_pos_val + false_neg_val) > 0 else 0
    recall_neg_val = true_neg_val / (true_neg_val + false_pos_val) if (true_neg_val + false_pos_val) > 0 else 0

    print(f'Validation Loss: {val_loss/len(val_loader):.4f}, '
          f'Validation Recall (Pos): {recall_pos_val:.4f}, Validation Recall (Neg): {recall_neg_val:.4f}')
    
train_model(model_squeeze, train_loader, val_loader, criterion, optimizer, num_epochs=15)

def test_model(model, test_loader):
    model.eval()
    all_labels = []
    all_predictions = []

    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)

            _, predicted = torch.max(outputs, 1)
            all_labels.extend(labels.cpu().numpy())
            all_predictions.extend(predicted.cpu().numpy())

    # Compute precision, recall, and F1 score
    precision = precision_score(all_labels, all_predictions, average='binary')
    recall = recall_score(all_labels, all_predictions, average='binary')
    f1 = f1_score(all_labels, all_predictions, average='binary')

    # Compute accuracy
    accuracy = sum(np.array(all_predictions) == np.array(all_labels)) / len(all_labels)

    # Print out the metrics
    print(f'Test Accuracy: {100 * accuracy:.2f}%')
    print(f'Precision: {precision:.4f}')
    print(f'Recall: {recall:.4f}')
    print(f'F1 Score: {f1:.4f}')

all_true_labels = []
all_pred_probs = []

test_model(model_squeeze, test_loader)

model_squeeze.eval()

with torch.no_grad():
    for images, labels in test_loader:
        images, labels = images.to(device), labels.to(device)
        
        outputs = model_squeeze(images)
        probabilities = torch.softmax(outputs, dim=1)[:, 1]

        all_true_labels.extend(labels.cpu().numpy())
        all_pred_probs.extend(probabilities.cpu().numpy())

print(f"Sample True Labels: {all_true_labels[:10]}")
print(f"Sample Predicted Labels: {all_pred_probs[:10]}")

with open("squeeze_labels_data.pkl", "wb") as f:
    pickle.dump({"true_labels": all_true_labels, "pred_labels": all_pred_probs}, f)