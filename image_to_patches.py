import numpy as np
from PIL import Image
import torchvision.transforms as transforms
import random


class PatchAugmentation(object):
    def __init__(self, patch_size, num_transformations):
        self.transformations = [
            transforms.ColorJitter(brightness=0.1, hue=0.1),
            transforms.GaussianBlur(kernel_size=(1, 5), sigma=(0.1, 2)),
            transforms.RandomAdjustSharpness(sharpness_factor=2),
            transforms.RandomPosterize(bits=3),
            # transforms.RandomAutocontrast(),
            # transforms.AugMix(),
            # transforms.RandAugment(num_ops=1, magnitude=1),
            ]
        self.patch_size = patch_size
        self.num_transformations = num_transformations

    def resize_image(self):
        width, height = self.image.size
        patch_width, patch_height = self.patch_size
        
        # Calculate the new size that is divisible by the patch size
        new_width = (width // patch_width) * patch_width
        new_height = (height // patch_height) * patch_height
        
        # Resize the image
        self.resized_image = self.image.resize((new_width, new_height))
        
        return self.resized_image, (width, height)

    def image_to_patches(self):
        width, height = self.resized_image.size
        patch_width, patch_height = self.patch_size
        patches = []
        
        for y in range(0, height, patch_height):
            for x in range(0, width, patch_width):
                patch = self.image.crop((x, y, x + patch_width, y + patch_height))
                # transformed_patch = transform(patch)
                transformed_patch = self.apply_random_transformations(patch)

                patches.append(transformed_patch)
        
        return patches

    def patches_to_image(self):
        width, height = self.resized_image.size
        patch_width, patch_height = self.patch_size
        rows = height // patch_height
        cols = width // patch_width
        image = Image.new('RGB', (width, height))
        
        for i, (patch, rv) in enumerate(self.patches):
            row = i // cols
            col = i % cols
            x = col * patch_width
            y = row * patch_height
            image.paste(patch, (x, y))
        
        return image

    def apply_random_transformations(self, image):
        transformed_image = image.copy()
        selected_transformations = random.sample(self.transformations, self.num_transformations)
        random.shuffle(selected_transformations)

        composed_transforms = transforms.Compose(selected_transformations)
        self.transformed_image = composed_transforms(transformed_image)

        return self.transformed_image

    def __call__(self, image):
        self.image = image

        # Resize the image to ensure divisibility by the patch size
        self.resized_image, self.original_size = self.resize_image()

        # Convert resized image to patches and apply transformations on each patch
        self.patches = self.image_to_patches()

        # Convert patches back to image
        self.reconstructed_image = self.patches_to_image()
        return self.reconstructed_image

    def __repr__(self):
        return "patch augmentation"
#######################################################################

# Example usage

# image_path = '/home/alij/dino_alijavidani/dog.jpg'
# image = Image.open(image_path)

# patch_size = (16, 16)
# num_transformations = 3

# p = PatchAugmentation(patch_size, num_transformations)
# ali = p(image)

# t= transforms.Compose([
#     # transforms.CenterCrop(size=(150,180)),
#     transforms.ColorJitter(brightness=5, contrast=10),
#     transforms.RandomHorizontalFlip(),
#     transforms.RandomResizedCrop(size=(150,180)),
#     p,
# ])

# ali2, rv = t(image)
# print(rv)
# ali2.save('ali2.jpg')
# ali.save('ali.jpg')