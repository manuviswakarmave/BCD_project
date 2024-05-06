import torch
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from joblib import dump
from torchvision.datasets import ImageFolder
from torchvision.transforms import transforms
from torch.utils.data import DataLoader
import torchvision.models as models

# Define transformation to preprocess the input image
transform = transforms.Compose([
    transforms.Resize((224, 224)),  # Resize to match the input size of ResNet
    transforms.ToTensor(),  # Convert to tensor
])

# Load and preprocess the data
train_data = ImageFolder('segmented_masks/train', transform=transform)
test_data = ImageFolder('segmented_masks/test', transform=transform)

# Define data loaders
train_loader = DataLoader(train_data, batch_size=32, shuffle=True)
test_loader = DataLoader(test_data, batch_size=32, shuffle=False)

# Load pre-trained ResNet model
resnet = models.resnet18(pretrained=True)
# Remove the last layer (classification layer)
resnet = torch.nn.Sequential(*list(resnet.children())[:-1])
# Set model to evaluation mode
resnet.eval()


# Extract features from images
def extract_features(loader, model):
    feature_list = []
    label_list = []
    for images, labels in loader:
        with torch.no_grad():
            features = model(images).squeeze().numpy()
        feature_list.append(features)
        label_list.append(labels.numpy())
    return np.concatenate(feature_list), np.concatenate(label_list)


train_features, train_labels = extract_features(train_loader, resnet)
test_features, test_labels = extract_features(test_loader, resnet)

# Initialize and train Random Forest classifier
random_forest = RandomForestClassifier(n_estimators=100)
random_forest.fit(train_features, train_labels)

# Evaluate the model
predictions = random_forest.predict(test_features)
accuracy = accuracy_score(test_labels, predictions)
print(f'Accuracy: {accuracy}')

# Save the trained Random Forest model
dump(random_forest, 'random_forest_model1.pkl')
