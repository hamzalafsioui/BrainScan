import os
import cv2
import numpy as np
from pathlib import Path
from utility import load_image_safe, get_classes, is_valid_image

class DataLoader:
   
    
    def __init__(self, data_path, target_size=(224, 224)):
        self.data_path = data_path
        self.target_size = target_size
        self.classes = get_classes(data_path)
        self.class_to_idx = {cls: idx for idx, cls in enumerate(self.classes)}
        
        print(f"\nFound classes: {self.classes}")
        print(f"Target image size: {target_size}")
    
    def load_all_images(self):
        
        images = []
        labels = []
        
        print("\nLoading images...")
        
        # Loop through each class folder
        for class_name in self.classes:
            class_path = os.path.join(self.data_path, class_name)
            class_idx = self.class_to_idx[class_name]
            
            # Get all files in class folder
            file_list = os.listdir(class_path)
            valid_images = 0
            
            for file_name in file_list:
                file_path = os.path.join(class_path, file_name)
                
                # Skip if not valid image
                if not is_valid_image(file_path):
                    continue
                
                # Load image
                image = load_image_safe(file_path, self.target_size)
                
                if image is not None:
                    images.append(image)
                    labels.append(class_idx)
                    valid_images += 1
            
            print(f"  Class '{class_name}': {valid_images} images loaded")
        
        print(f"\nTotal images loaded: {len(images)}")
        
        return images, labels
    
    def to_numpy_arrays(self, images, labels):
        
        X = np.array(images, dtype=np.float32)
        y = np.array(labels, dtype=np.int32)
        
        print(f"\nNumpy arrays created:")
        print(f"  X shape: {X.shape}")
        print(f"  y shape: {y.shape}")
        
        return X, y