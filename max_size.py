
from torchvision import datasets

# Define the folder containing the images
folder_path = "Dataset/benign"

# Create a dataset from the folder
dataset = datasets.ImageFolder(folder_path)

# Initialize variables to store information about the image with maximum size
max_size = 0
max_size_image_path = None

# Iterate through the dataset to find the image with maximum size
for image_path, _ in dataset.imgs:
    # Open the image using torchvision and get its size
    image = datasets.folder.default_loader(image_path)
    image_size = max(image.size)

    # Update max_size and max_size_image_path if current image size is larger
    if image_size > max_size:
        max_size = image_size
        max_size_image_path = image_path

# Print the filename of the image with maximum size
if max_size_image_path is not None:
    print("Image with maximum size:", max_size_image_path)
else:
    print("No images found in the folder.")
