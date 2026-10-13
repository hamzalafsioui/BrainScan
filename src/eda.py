import matplotlib.pyplot as plt
import numpy as np
from collections import Counter

def plot_class_distribution(labels, class_names):
    """
    Show bar chart of images per class
    """
    # Count images in each class
    label_counts = Counter(labels)
    
    # Sort by class name
    classes = sorted(label_counts.keys())
    counts = [label_counts[c] for c in classes]
    class_labels = [class_names[c] for c in classes]
    
    # Create bar chart
    plt.figure(figsize=(8, 5))
    plt.bar(class_labels, counts, color=['#FF6B6B', '#4ECDC4'])
    plt.title('Dataset Distribution', fontsize=16, fontweight='bold')
    plt.xlabel('Class', fontsize=12)
    plt.ylabel('Number of Images', fontsize=12)
    plt.grid(axis='y', alpha=0.3)
    
    # Add count labels on bars
    for i, count in enumerate(counts):
        plt.text(i, count + 5, str(count), ha='center', fontweight='bold')
    
    plt.tight_layout()
    plt.show()

def plot_sample_images(images, labels, class_names, num_samples=4):
    """
    Display sample images from each class
    """
    unique_classes = sorted(set(labels))
    
    fig, axes = plt.subplots(len(unique_classes), num_samples, 
                              figsize=(12, 8))
    
    for class_idx, class_id in enumerate(unique_classes):
        # Get indices of images from this class
        class_indices = [i for i, label in enumerate(labels) if label == class_id]
        
        # Sample up to num_samples
        sample_indices = class_indices[:num_samples]
        
        for sample_num, img_idx in enumerate(sample_indices):
            ax = axes[class_idx, sample_num]
            
            # Get image (convert from BGR to RGB for display)
            image = images[img_idx]
            image_rgb = image[..., ::-1] / 255.0  # BGR to RGB and normalize
            
            ax.imshow(image_rgb)
            ax.set_title(f"{class_names[class_id]}")
            ax.axis('off')
    
    plt.tight_layout()
    plt.show()

def check_class_balance(labels, class_names):
    
    from collections import Counter
    
    label_counts = Counter(labels)
    total = len(labels)
    
    print("\n" + "="*50)
    print("CLASS BALANCE")
    print("="*50)
    
    for class_id in sorted(label_counts.keys()):
        count = label_counts[class_id]
        percentage = (count / total) * 100
        print(f"{class_names[class_id]}: {count} images ({percentage:.1f}%)")
    
    print("="*50 + "\n")
    
    # Check if balanced
    counts = list(label_counts.values())
    if max(counts) / min(counts) > 1.5:
        print("Dataset is imbalanced \n")
    else:
        print("Dataset is fairly balanced.\n")