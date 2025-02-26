"""
Example script demonstrating synthetic data generation for underrepresented classes.
"""

import os
import sys
import pandas as pd
import numpy as np
import cv2
import matplotlib.pyplot as plt
from pathlib import Path
import random
import shutil

# Add parent directory to path so we can import our package
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from data_preprocessing.underrepresented_classes import (
    find_underrepresented_classes,
    generate_synthetic_data,
    visualize_class_distribution,
    report_class_balance_issues
)


def create_sample_dataset(base_dir, class_counts=None):
    """
    Create a sample dataset with artificial class imbalance for demonstration purposes.
    
    Args:
        base_dir: Directory where to create the dataset
        class_counts: Dictionary mapping class IDs to number of samples
        
    Returns:
        DataFrame with bounding box information
    """
    if class_counts is None:
        class_counts = {0: 100, 1: 80, 2: 20, 3: 5, 4: 50}  # Intentionally imbalanced
    
    # Create directories
    os.makedirs(os.path.join(base_dir, 'images'), exist_ok=True)
    os.makedirs(os.path.join(base_dir, 'labels'), exist_ok=True)
    
    # Class names for reference
    class_names = {
        0: "Fish", 1: "Coral", 2: "Starfish", 3: "Crab", 4: "Turtle"
    }
    
    # Generate sample data
    all_boxes = []
    
    for class_id, count in class_counts.items():
        for i in range(count):
            # Create a simple image with a colored rectangle
            img_size = (640, 480)
            img = np.zeros((img_size[1], img_size[0], 3), dtype=np.uint8)
            
            # Add background texture
            for _ in range(1000):
                x = random.randint(0, img_size[0]-1)
                y = random.randint(0, img_size[1]-1)
                color = random.randint(30, 80)
                cv2.circle(img, (x, y), random.randint(1, 3), 
                          (color, color, color+20), -1)
            
            # Class-specific colors
            colors = {
                0: (50, 50, 200),  # Fish: red
                1: (50, 200, 50),  # Coral: green
                2: (200, 50, 200),  # Starfish: purple
                3: (200, 150, 50),  # Crab: blue
                4: (50, 200, 200)   # Turtle: yellow
            }
            
            # Generate random bounding box
            box_w = random.randint(100, 200)
            box_h = random.randint(100, 200)
            x_min = random.randint(50, img_size[0] - box_w - 50)
            y_min = random.randint(50, img_size[1] - box_h - 50)
            x_max = x_min + box_w
            y_max = y_min + box_h
            
            # Draw the object
            cv2.rectangle(img, (x_min, y_min), (x_max, y_max), colors[class_id], -1)
            
            # Add some texture to the object
            for _ in range(100):
                x = random.randint(x_min, x_max)
                y = random.randint(y_min, y_max)
                r, g, b = colors[class_id]
                cv2.circle(img, (x, y), random.randint(1, 5), 
                          (min(r+50, 255), min(g+50, 255), min(b+50, 255)), -1)
            
            # Create filename
            filename = f"sample_{class_names[class_id].lower()}_{i:03d}.jpg"
            img_path = os.path.join(base_dir, 'images', filename)
            
            # Save the image
            cv2.imwrite(img_path, img)
            
            # Create a YOLO format label file (normalized coordinates)
            label_path = os.path.join(base_dir, 'labels', f"{os.path.splitext(filename)[0]}.txt")
            with open(label_path, 'w') as f:
                # class_id center_x center_y width height (normalized)
                center_x = (x_min + x_max) / 2 / img_size[0]
                center_y = (y_min + y_max) / 2 / img_size[1]
                norm_width = box_w / img_size[0]
                norm_height = box_h / img_size[1]
                f.write(f"{class_id} {center_x:.6f} {center_y:.6f} {norm_width:.6f} {norm_height:.6f}\n")
            
            # Add to bounding box DataFrame
            all_boxes.append({
                'filename': filename,
                'class_id': class_id,
                'xmin': x_min,
                'ymin': y_min,
                'xmax': x_max,
                'ymax': y_max,
                'width': box_w,
                'height': box_h,
                'img_width': img_size[0],
                'img_height': img_size[1]
            })
    
    return pd.DataFrame(all_boxes), class_names


def visualize_samples(df, images_dir, class_names, num_samples=5):
    """Visualize random samples from each class."""
    unique_classes = df['class_id'].unique()
    
    plt.figure(figsize=(15, 3 * len(unique_classes)))
    
    for i, class_id in enumerate(unique_classes):
        class_samples = df[df['class_id'] == class_id]
        sample_size = min(num_samples, len(class_samples))
        
        if sample_size == 0:
            continue
            
        sample_rows = class_samples.sample(sample_size)
        
        for j, (_, row) in enumerate(sample_rows.iterrows()):
            plt.subplot(len(unique_classes), num_samples, i*num_samples + j + 1)
            
            # Load image
            img_path = os.path.join(images_dir, row['filename'])
            img = cv2.imread(img_path)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            
            # Draw bounding box
            cv2.rectangle(img, (row['xmin'], row['ymin']), (row['xmax'], row['ymax']), (0, 255, 0), 2)
            
            plt.imshow(img)
            plt.title(f"{class_names[row['class_id']]}")
            plt.axis('off')
    
    plt.tight_layout()
    plt.savefig(os.path.join(os.path.dirname(images_dir), 'sample_visualization.png'))
    plt.close()


def main():
    print("=== Ocean CV Bench: Synthetic Data Generation Demo ===\n")
    
    # Create a base directory for our example
    base_dir = os.path.join(os.path.dirname(__file__), 'synthetic_data_demo')
    os.makedirs(base_dir, exist_ok=True)
    
    # Create sample dataset with imbalanced classes
    print("Creating sample dataset with imbalanced classes...")
    df_boxes, class_names = create_sample_dataset(base_dir)
    
    # Visualize some samples
    print("Visualizing sample images...")
    visualize_samples(df_boxes, os.path.join(base_dir, 'images'), class_names)
    
    # Analyze class distribution
    print("\nAnalyzing class distribution...")
    report = report_class_balance_issues(df_boxes, threshold=30, class_names=class_names)
    
    print(f"Total samples: {report['total_boxes']}")
    print("Class counts:")
    for class_id, count in report['class_counts'].items():
        print(f"  {class_names[int(class_id)]}: {count}")
    
    print("\nUnderrepresented classes:")
    for class_id in report['underrepresented_classes']:
        print(f"  {class_names[int(class_id)]}: {report['class_counts'][class_id]} samples")
    
    # Visualize class distribution
    viz_path = os.path.join(base_dir, 'class_distribution.png')
    visualize_class_distribution(df_boxes, class_names=class_names, output_path=viz_path)
    print(f"Class distribution chart saved to {viz_path}")
    
    # Generate synthetic data for the most underrepresented class
    if report['underrepresented_classes']:
        target_class = report['underrepresented_classes'][0]
        target_count = 20  # Generate 20 additional samples
        
        print(f"\nGenerating {target_count} synthetic samples for {class_names[target_class]}...")
        
        # Create a directory for synthetic data
        synthetic_dir = os.path.join(base_dir, 'synthetic')
        
        # Generate the synthetic data
        results = generate_synthetic_data(
            class_id=target_class,
            count_needed=target_count,
            df_bboxes=df_boxes,
            images_dir=os.path.join(base_dir, 'images'),
            output_dir=synthetic_dir,
            class_names=class_names,
            augmentation_intensity='medium'
        )
        
        # Display results
        print(f"\nGenerated {len(results['generated_images'])} synthetic images")
        print(f"Techniques used: {', '.join(results['techniques_used'])}")
        
        # Create a new combined dataset with both original and synthetic data
        combined_dir = os.path.join(base_dir, 'combined')
        os.makedirs(os.path.join(combined_dir, 'images'), exist_ok=True)
        os.makedirs(os.path.join(combined_dir, 'labels'), exist_ok=True)
        
        # Copy all original data
        print("\nCreating combined dataset with original and synthetic data...")
        for filename in os.listdir(os.path.join(base_dir, 'images')):
            shutil.copy2(
                os.path.join(base_dir, 'images', filename),
                os.path.join(combined_dir, 'images', filename)
            )
            
            # Copy corresponding label file if it exists
            label_name = os.path.splitext(filename)[0] + '.txt'
            label_path = os.path.join(base_dir, 'labels', label_name)
            if os.path.exists(label_path):
                shutil.copy2(
                    label_path,
                    os.path.join(combined_dir, 'labels', label_name)
                )
        
        # Copy synthetic data
        for img_path in results['generated_images']:
            filename = os.path.basename(img_path)
            shutil.copy2(
                img_path,
                os.path.join(combined_dir, 'images', filename)
            )
        
        for label_path in results['generated_labels']:
            label_name = os.path.basename(label_path)
            shutil.copy2(
                label_path,
                os.path.join(combined_dir, 'labels', label_name)
            )
        
        # Analyze the combined dataset
        print("\nAnalyzing combined dataset after adding synthetic samples...")
        
        # Re-build the bounding box DataFrame for the combined dataset
        combined_boxes = []
        
        # Process all label files in the combined directory
        for label_file in os.listdir(os.path.join(combined_dir, 'labels')):
            if not label_file.endswith('.txt'):
                continue
                
            base_name = os.path.splitext(label_file)[0]
            img_file = base_name + '.jpg'
            
            # Skip if image file doesn't exist
            if not os.path.exists(os.path.join(combined_dir, 'images', img_file)):
                continue
            
            # Read label file
            with open(os.path.join(combined_dir, 'labels', label_file), 'r') as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) >= 5:
                        class_id = int(parts[0])
                        center_x = float(parts[1])
                        center_y = float(parts[2])
                        width = float(parts[3])
                        height = float(parts[4])
                        
                        # Convert normalized YOLO format to absolute coordinates
                        img_width, img_height = 640, 480
                        xmin = int((center_x - width/2) * img_width)
                        ymin = int((center_y - height/2) * img_height)
                        xmax = int((center_x + width/2) * img_width)
                        ymax = int((center_y + height/2) * img_height)
                        
                        combined_boxes.append({
                            'filename': img_file,
                            'class_id': class_id,
                            'xmin': xmin,
                            'ymin': ymin,
                            'xmax': xmax,
                            'ymax': ymax,
                            'width': xmax - xmin,
                            'height': ymax - ymin,
                            'img_width': img_width,
                            'img_height': img_height
                        })
                        
        # Create DataFrame with all boxes
        df_combined = pd.DataFrame(combined_boxes)
        
        # Show new class distribution
        viz_path = os.path.join(base_dir, 'combined_class_distribution.png')
        visualize_class_distribution(df_combined, class_names=class_names, output_path=viz_path)
        print(f"Updated class distribution chart saved to {viz_path}")
        
        # Report the new distribution
        updated_report = report_class_balance_issues(df_combined, threshold=30, class_names=class_names)
        
        print("\nUpdated class counts:")
        for class_id, count in updated_report['class_counts'].items():
            print(f"  {class_names[int(class_id)]}: {count}")
        
    print("\nDemo completed! Check the output files in the 'synthetic_data_demo' directory.")


if __name__ == "__main__":
    main()