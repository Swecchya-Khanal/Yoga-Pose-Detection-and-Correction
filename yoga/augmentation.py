import cv2
import os
from albumentations import (
    Compose, HorizontalFlip, ShiftScaleRotate, RandomBrightnessContrast, 
    HueSaturationValue, ElasticTransform, CoarseDropout, GaussianBlur, Perspective
)

def augment_image(image, output_dir, augmentations, base_filename, num_augmentations=10):
    """
    Apply augmentations to a single image and save the results.
    Args:
        image: Input image (as a numpy array).
        output_dir: Directory to save augmented images.
        augmentations: Albumentations Compose object.
        base_filename: Base name of the input image for naming augmented images.
        num_augmentations: Number of augmented images to create.
    """
    os.makedirs(output_dir, exist_ok=True)
    for i in range(num_augmentations):
        augmented = augmentations(image=image)
        augmented_image = augmented["image"]
        filename = f"{base_filename}_aug_{i}.png"
        filepath = os.path.join(output_dir, filename)
        cv2.imwrite(filepath, augmented_image)
    print(f"{num_augmentations} augmented images saved for {base_filename}")


augmentations = Compose([
    HorizontalFlip(p=0.5),
    ShiftScaleRotate(shift_limit=0.1, scale_limit=0.2, rotate_limit=30, p=0.8),
    RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.7),
    HueSaturationValue(hue_shift_limit=10, sat_shift_limit=15, val_shift_limit=10, p=0.6),
    ElasticTransform(alpha=1, sigma=50, p=0.5),
    CoarseDropout(max_holes=8, max_height=30, max_width=30, min_holes=1, p=0.4),
    GaussianBlur(blur_limit=(3, 7), p=0.3),
    Perspective(scale=(0.05, 0.1), p=0.4),
], p=1.0)


input_folder = "C:/Users/inoug/Desktop/Dataset/mountain_pose" 
output_folder = "C:/Users/inoug/Desktop/Dataset/mountain"  
num_augmentations_per_image = 20  


for filename in os.listdir(input_folder):
    if filename.lower().endswith((".png", ".jpg", ".jpeg")):  
        image_path = os.path.join(input_folder, filename)
        image = cv2.imread(image_path)
        if image is not None:
            base_filename, _ = os.path.splitext(filename)
            augment_image(image, output_folder, augmentations, base_filename, num_augmentations=num_augmentations_per_image)
        else:
            print(f"Failed to load image: {filename}")
