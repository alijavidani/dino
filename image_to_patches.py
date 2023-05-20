import numpy as np
from PIL import Image
import torchvision.transforms as transforms
import random


def resize_image(image, patch_size):
    width, height = image.size
    patch_width, patch_height = patch_size
    
    # Calculate the new size that is divisible by the patch size
    new_width = (width // patch_width) * patch_width
    new_height = (height // patch_height) * patch_height
    
    # Resize the image
    resized_image = image.resize((new_width, new_height))
    
    return resized_image, (width, height)


def image_to_patches(image, patch_size, transformations, num_transformations):
    width, height = image.size
    patch_width, patch_height = patch_size
    patches = []
    
    for y in range(0, height, patch_height):
        for x in range(0, width, patch_width):
            patch = image.crop((x, y, x + patch_width, y + patch_height))
            # transformed_patch = transform(patch)
            transformed_patch = apply_random_transformations(patch, transformations, num_transformations)

            patches.append(transformed_patch)
    
    return patches


def patches_to_image(patches, image_size):
    width, height = image_size
    patch_width, patch_height = patches[0].size
    rows = height // patch_height
    cols = width // patch_width
    image = Image.new('RGB', (width, height))
    
    for i, patch in enumerate(patches):
        row = i // cols
        col = i % cols
        x = col * patch_width
        y = row * patch_height
        image.paste(patch, (x, y))
    
    return image


def apply_random_transformations(image, transformations, num_transformations):
    transformed_image = image.copy()
    selected_transformations = random.sample(transformations, num_transformations)
    random.shuffle(selected_transformations)

    composed_transforms = transforms.Compose(selected_transformations)
    transformed_image = composed_transforms(transformed_image)

    return transformed_image


def patch_augmentation(image, patch_size, transformations, num_transformations):
    # Resize the image to ensure divisibility by the patch size
    resized_image, original_size = resize_image(image, patch_size)

    # Convert resized image to patches
    patches = image_to_patches(resized_image, patch_size, transformations, num_transformations)

    # Convert patches back to image
    reconstructed_image = patches_to_image(patches, original_size)
    return reconstructed_image
#######################################################################

# # Example usage
# image_path = 'c:/Users/alija/Desktop/dog.jpg'
# patch_size = (16, 16)

# # Open the image
# image = Image.open(image_path)
# num_transformations = 2

# patch_augmentation_transformations = [transforms.ColorJitter(brightness=0.1, hue=0.1),
#     transforms.GaussianBlur(kernel_size=(5, 9), sigma=(0.1, 5)),
#     transforms.RandomAutocontrast(),
#     transforms.RandomAdjustSharpness(sharpness_factor=2),
#     transforms.RandomPosterize(bits=3),
#     transforms.AugMix(),
#     transforms.RandAugment(num_ops=1, magnitude=1),]

# reconstructed_image = patch_augmentation(image, patch_size, patch_augmentation_transformations, num_transformations)

# # Display the original and reconstructed images
# # image.show()
# reconstructed_image.show()

