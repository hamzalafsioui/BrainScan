import os
import cv2
import numpy as np
from pathlib import Path

VALID_EXTENSIONS = ('.jpeg', '.jpg', '.bmp', '.png')

def is_valid_image(filepath):
    return filepath.lower().endwith(VALID_EXTENSIONS)

def load_image_safe(file_path,target_size = (224,224)):

    try:

        image = cv2.imread(file_path)

        # check image loaded success
        if image is None:
            print("failed to load ",file_path)
            return None

        image_resized = cv2.resize(image,target_size)

        return image_resized
    except Exception as e:
        print("Error loading ",e)



def clean_dataset(data_path):
    deleted_count = 0

    for root,dirs,files in os.walk(data_path):
        for file in files:
            file_path = os.path.join(root,file)

            # if file is not valid image, delete it
            if not is_valid_image(file_path):
                try:
                    os.remove(file_path)
                    print(f"Deleted: {file_path}")
                    deleted_count +=1

                except Exception as e:
                    print(f"Error deleting {file_path}: {e}")

    return deleted_count


    