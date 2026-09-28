import os
import cv2
import albumentations as A
from tqdm import tqdm

# Define the augmentation pipeline
augmentation_pipeline = A.Compose([
    A.OneOf([
        A.RandomBrightnessContrast(p=0.5),
        A.HueSaturationValue(p=0.5),
        A.RGBShift(p=0.5),
    ], p=0.7),
    A.OneOf([
        A.HorizontalFlip(p=0.5),
        A.Rotate(limit=15, p=0.5),
        A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.1, rotate_limit=15, p=0.7),
    ], p=0.7),
    A.OneOf([
        A.GaussianBlur(p=0.5),
        A.GaussNoise(var_limit=(10.0, 50.0), p=0.5),
    ], p=0.5)
])

# Function to apply augmentation and save images
def augment_and_save_images(input_folder, target_count):
    for folder_name in os.listdir(input_folder):
        folder_path = os.path.join(input_folder, folder_name)
        
        if not os.path.isdir(folder_path):
            continue
        
        # Get all images in the folder
        images = [f for f in os.listdir(folder_path) if f.lower().endswith(('png', 'jpg', 'jpeg'))]
        current_count = len(images)
        
        if current_count >= target_count:
            continue  # Skip folders with enough images
        
        print(f"Augmenting images in folder: {folder_name} (current: {current_count}, target: {target_count})")
        
        # Loop through images to augment them
        for i in tqdm(range(target_count - current_count)):
            # Pick a random image from the folder
            random_image = images[i % current_count]
            image_path = os.path.join(folder_path, random_image)
            image = cv2.imread(image_path)
            
            if image is None:
                print(f"Skipping {random_image} as it could not be read.")
                continue
            
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            # Apply augmentation
            augmented = augmentation_pipeline(image=image)
            augmented_image = augmented['image']
            
            # Save the augmented image back to the folder
            new_image_name = f"aug_{i}_{random_image}"
            save_path = os.path.join(folder_path, new_image_name)
            augmented_image = cv2.cvtColor(augmented_image, cv2.COLOR_RGB2BGR)
            cv2.imwrite(save_path, augmented_image)
            print(f"Saved augmented image: {new_image_name}")

# Path to the dataset folder
dataset_folder = r"Dataset/v2/newisis"  # Replace with your actual dataset path
target_image_count = 550  # Set the target number of images per folder

# Apply augmentation
augment_and_save_images(dataset_folder, target_image_count)
