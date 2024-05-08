import torch
from torchvision.transforms import transforms
import torchvision.models as models
from joblib import load


# Load pre-trained ResNet model
resnet = models.resnet18(pretrained=True)
# Remove the last layer (classification layer)
resnet = torch.nn.Sequential(*list(resnet.children())[:-1])
# Set model to evaluation mode
resnet.eval()

# Load trained Random Forest model
random_forest = load('random_forest_model1.pkl')

# Define transformation to preprocess the input image
transform = transforms.Compose([
    transforms.Resize((224, 224)),  # Resize to match the input size of ResNet
    transforms.ToTensor(),  # Convert to tensor
])


def predict(image):
    # Load and preprocess the input image

    image = transform(image).unsqueeze(0)  # Add batch dimension

    # Extract features from the input image using ResNet
    with torch.no_grad():
        features = resnet(image).squeeze().numpy()

    # Make predictions using the trained Random Forest classifier
    prediction = random_forest.predict([features])

    return prediction[0]
