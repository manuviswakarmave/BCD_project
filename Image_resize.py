import os
from PIL import Image


def resize_or_pad_image(image_path, target_size=(1028, 565)):
    # Open the image
    image = Image.open(image_path)

    # Resize or pad the image to the target size
    new_image = Image.new("RGB", target_size, (0, 0, 0))
    new_image.paste(image, ((target_size[0] - image.width) // 2, (target_size[1] - image.height) // 2))

    return new_image


def process_images_in_folder(folder_path, output_folder_path):
    # Create the output folder if it doesn't exist
    os.makedirs(output_folder_path, exist_ok=True)

    # List all files in the input folder
    files = os.listdir(folder_path)

    # Process each image in the input folder
    for file in files:
        input_file_path = os.path.join(folder_path, file)
        output_file_path = os.path.join(output_folder_path, file)

        # Resize or pad the image
        new_image = resize_or_pad_image(input_file_path)

        # Save the processed image
        new_image.save(output_file_path)

from PIL import Image

def resize_image(image_data, target_size=(1028, 565)):
    """
    Resize or pad the image to the target size while maintaining its aspect ratio.
    If the image is smaller than the target size, it will be padded with black color.
    If the image is larger than the target size, it will be resized to fit within the target size.

    Args:
    - image_data (PIL.Image): The input image data.
    - target_size (tuple): Target size (width, height) for the output image.

    Returns:
    - new_image (PIL.Image): Resized or padded image.
    """
    # Get the width and height of the input image
    width, height = image_data.size

    # Create a new image with the target size and black background
    new_image = Image.new("RGB", target_size, (0, 0, 0))

    # Calculate the position to paste the input image
    paste_x = (target_size[0] - width) // 2
    paste_y = (target_size[1] - height) // 2

    # Paste the input image onto the new image
    new_image.paste(image_data, (paste_x, paste_y))

    return new_image



# Example usage:
# input_folder_path = "segmented_masks/train/benign"
# output_folder_path = "D:\\Development\\Python Projects\\Demo\\BCD_Project\\new_mask\\train\\benign"
#
# process_images_in_folder(input_folder_path, output_folder_path)
