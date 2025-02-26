"""Functions to identify and handle underrepresented classes."""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import random
from pathlib import Path


def find_underrepresented_classes(df_bboxes, threshold=50):
    """
    Identifies classes with fewer than threshold samples.
    
    Args:
        df_bboxes: DataFrame with bounding box information including 'class_id'
        threshold: Minimum number of samples required per class
        
    Returns:
        List of underrepresented class IDs
    """
    if 'class_id' not in df_bboxes.columns:
        raise ValueError("DataFrame must contain a 'class_id' column")
        
    # Count occurrences of each class
    class_counts = df_bboxes['class_id'].value_counts()
    
    # Find classes below threshold
    underrepresented = class_counts[class_counts < threshold].index.tolist()
    
    return underrepresented


def generate_synthetic_data(class_id, count_needed, df_bboxes=None, images_dir=None, output_dir=None, 
                           existing_samples=None, class_names=None, augmentation_intensity='medium'):
    """
    Generates synthetic data for underrepresented classes using augmentation techniques.
    
    Args:
        class_id: ID of the class that needs more samples
        count_needed: Number of additional samples to generate
        df_bboxes: DataFrame with bounding box information for source images
        images_dir: Directory containing the original images
        output_dir: Directory where synthetic images will be saved
        existing_samples: Optional list of existing image paths containing the target class
        class_names: Dict mapping class IDs to human-readable names
        augmentation_intensity: Level of augmentation ('mild', 'medium', 'aggressive')
        
    Returns:
        Dict with paths to generated images and their annotations
    """
    import os
    import cv2
    import numpy as np
    import random
    import uuid
    from pathlib import Path
    from datetime import datetime
    import shutil
    
    try:
        # Import augmentation functions
        from .augmentation import augment_biofouling, augment_shear_perspective, augment_camera_distance
    except ImportError:
        print("Warning: Could not import augmentation functions from sibling module.")
        # Define simple fallback functions if imports fail
        def augment_biofouling(img, _): return img
        def augment_shear_perspective(img): return img
        def augment_camera_distance(img): return img
    
    # Create output directory if specified but doesn't exist
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        os.makedirs(os.path.join(output_dir, 'images'), exist_ok=True)
        os.makedirs(os.path.join(output_dir, 'labels'), exist_ok=True)
    else:
        # Use a temporary directory if not specified
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = f"synthetic_data_{class_id}_{timestamp}"
        os.makedirs(output_dir, exist_ok=True)
        os.makedirs(os.path.join(output_dir, 'images'), exist_ok=True)
        os.makedirs(os.path.join(output_dir, 'labels'), exist_ok=True)
    
    # Initialize results dictionary
    results = {
        'generated_images': [],
        'generated_labels': [],
        'class_id': class_id,
        'count_requested': count_needed,
        'count_generated': 0,
        'techniques_used': []
    }
    
    # Display class name if available
    class_name = class_names[class_id] if class_names and class_id in class_names else f"Class {class_id}"
    print(f"Generating {count_needed} synthetic samples for {class_name} (ID: {class_id})")
    
    # 1. Find source samples containing the target class
    source_samples = []
    
    # If existing_samples is provided, use those
    if existing_samples:
        source_samples = existing_samples
        print(f"Using {len(existing_samples)} provided source samples for synthesis")
    
    # Otherwise, if both df_bboxes and images_dir are provided, extract source samples
    elif df_bboxes is not None and images_dir:
        # Filter for target class
        class_samples = df_bboxes[df_bboxes['class_id'] == class_id]
        if 'filename' in class_samples.columns:
            # Get unique filenames containing this class
            unique_files = class_samples['filename'].unique()
            source_samples = [os.path.join(images_dir, filename) for filename in unique_files]
            print(f"Found {len(source_samples)} source images containing class {class_id}")
        else:
            print("Warning: 'filename' column not found in df_bboxes")
            return results
    else:
        print("Error: Must provide either existing_samples or both df_bboxes and images_dir")
        return results
    
    # If no source samples found, exit
    if not source_samples:
        print(f"Error: No source samples found for class {class_id}")
        return results
    
    # Set augmentation parameters based on intensity
    intensity_params = {
        'mild': {
            'rotation_range': (-10, 10),
            'scale_range': (0.9, 1.1),
            'brightness_range': (0.9, 1.1),
            'contrast_range': (0.9, 1.1),
            'noise_prob': 0.2,
            'blur_prob': 0.2,
            'colorshift_prob': 0.2
        },
        'medium': {
            'rotation_range': (-20, 20),
            'scale_range': (0.8, 1.2),
            'brightness_range': (0.8, 1.2),
            'contrast_range': (0.8, 1.2),
            'noise_prob': 0.4,
            'blur_prob': 0.3,
            'colorshift_prob': 0.4
        },
        'aggressive': {
            'rotation_range': (-30, 30),
            'scale_range': (0.7, 1.3),
            'brightness_range': (0.7, 1.3),
            'contrast_range': (0.7, 1.3),
            'noise_prob': 0.6,
            'blur_prob': 0.4,
            'colorshift_prob': 0.6
        }
    }
    
    # Use medium intensity if specified intensity is not found
    params = intensity_params.get(augmentation_intensity, intensity_params['medium'])
    
    # Define augmentation functions
    def rotate_image(image, angle):
        """Rotate image by specified angle."""
        h, w = image.shape[:2]
        center = (w // 2, h // 2)
        M = cv2.getRotationMatrix2D(center, angle, 1.0)
        return cv2.warpAffine(image, M, (w, h), borderMode=cv2.BORDER_REPLICATE)
    
    def adjust_brightness_contrast(image, alpha=1.0, beta=0):
        """Adjust brightness (beta) and contrast (alpha)."""
        return cv2.convertScaleAbs(image, alpha=alpha, beta=beta)
    
    def add_noise(image, intensity=0.05):
        """Add random noise to the image."""
        noise = np.random.normal(0, intensity * 255, image.shape).astype(np.int16)
        noisy_img = cv2.add(image, noise.astype(np.int8))
        return np.clip(noisy_img, 0, 255).astype(np.uint8)
    
    def color_shift(image):
        """Randomly shift colors in the image."""
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        h, s, v = cv2.split(hsv)
        h = (h + random.randint(-20, 20)) % 180
        s = np.clip(s * random.uniform(0.8, 1.2), 0, 255).astype(np.uint8)
        hsv = cv2.merge([h, s, v])
        return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    
    def horizontal_flip(image):
        """Flip image horizontally."""
        return cv2.flip(image, 1)
    
    def apply_blur(image):
        """Apply slight Gaussian blur."""
        kernel_size = random.choice([3, 5, 7])
        return cv2.GaussianBlur(image, (kernel_size, kernel_size), 0)
    
    # 2. Generate synthetic images using various techniques
    generated_count = 0
    techniques_used = set()
    
    # Keep track of augmentation operations applied to each source
    source_usage_count = {sample: 0 for sample in source_samples}
    
    # Try to distribute generation evenly across source samples
    while generated_count < count_needed:
        # Select the least used source sample
        source_path = min(source_usage_count.items(), key=lambda x: x[1])[0]
        
        # Load the original image
        try:
            img = cv2.imread(source_path)
            if img is None:
                print(f"Warning: Could not read {source_path}, skipping.")
                # Remove from the source list
                source_usage_count.pop(source_path)
                if not source_usage_count:
                    print("Error: No valid source images left.")
                    break
                continue
        except Exception as e:
            print(f"Error loading {source_path}: {e}")
            source_usage_count.pop(source_path)
            if not source_usage_count:
                print("Error: No valid source images left.")
                break
            continue
        
        # Update usage count for this source
        source_usage_count[source_path] += 1
        
        # Create a unique filename for the synthetic image
        base_name = os.path.splitext(os.path.basename(source_path))[0]
        unique_id = str(uuid.uuid4())[:8]
        synthetic_filename = f"{base_name}_syn_{unique_id}.jpg"
        output_path = os.path.join(output_dir, 'images', synthetic_filename)
        
        # Start with the original image
        synthetic_img = img.copy()
        augmentations_applied = []
        
        # Apply random augmentations based on the parameters
        
        # 1. Random rotation
        if random.random() > 0.5:
            angle = random.uniform(params['rotation_range'][0], params['rotation_range'][1])
            synthetic_img = rotate_image(synthetic_img, angle)
            augmentations_applied.append(f"rotation_{angle:.1f}")
        
        # 2. Random brightness/contrast adjustment
        if random.random() > 0.3:
            alpha = random.uniform(params['contrast_range'][0], params['contrast_range'][1])
            beta = random.uniform(-20, 20)  # Brightness adjustment
            synthetic_img = adjust_brightness_contrast(synthetic_img, alpha, beta)
            augmentations_applied.append(f"brightness_contrast_{alpha:.2f}_{beta:.1f}")
        
        # 3. Add noise
        if random.random() < params['noise_prob']:
            noise_intensity = random.uniform(0.01, 0.05)
            synthetic_img = add_noise(synthetic_img, noise_intensity)
            augmentations_applied.append(f"noise_{noise_intensity:.2f}")
        
        # 4. Color shift
        if random.random() < params['colorshift_prob']:
            synthetic_img = color_shift(synthetic_img)
            augmentations_applied.append("color_shift")
        
        # 5. Horizontal flip
        if random.random() > 0.5:
            synthetic_img = horizontal_flip(synthetic_img)
            augmentations_applied.append("horizontal_flip")
        
        # 6. Apply blur
        if random.random() < params['blur_prob']:
            synthetic_img = apply_blur(synthetic_img)
            augmentations_applied.append("blur")
        
        # 7. Apply biofouling effect (if available)
        if 'augment_biofouling' in globals() and random.random() > 0.7:
            try:
                # Generate a simple texture if needed
                texture_path = os.path.join(output_dir, "temp_biofouling.jpg")
                if not os.path.exists(texture_path):
                    # Create a simple fouling texture
                    texture = np.zeros((512, 512, 3), dtype=np.uint8)
                    for _ in range(500):
                        x, y = random.randint(0, 511), random.randint(0, 511)
                        color = random.randint(100, 200)
                        cv2.circle(texture, (x, y), random.randint(5, 20), 
                                  (color, color-20, random.randint(50, 100)), -1)
                    cv2.imwrite(texture_path, texture)
                
                synthetic_img = augment_biofouling(synthetic_img, texture_path)
                augmentations_applied.append("biofouling")
            except Exception as e:
                print(f"Error applying biofouling: {e}")
        
        # 8. Apply shear/perspective (if available)
        if 'augment_shear_perspective' in globals() and random.random() > 0.6:
            try:
                synthetic_img = augment_shear_perspective(synthetic_img)
                augmentations_applied.append("shear_perspective")
            except Exception as e:
                print(f"Error applying shear: {e}")
        
        # 9. Apply camera distance effect (if available)
        if 'augment_camera_distance' in globals() and random.random() > 0.6:
            try:
                scale_factor = random.uniform(params['scale_range'][0], params['scale_range'][1])
                synthetic_img = augment_camera_distance(synthetic_img, scale=scale_factor)
                augmentations_applied.append(f"camera_distance_{scale_factor:.2f}")
            except Exception as e:
                print(f"Error applying camera distance effect: {e}")
        
        # Save the synthetic image
        try:
            cv2.imwrite(output_path, synthetic_img)
            results['generated_images'].append(output_path)
            techniques_used.update(augmentations_applied)
            
            # Copy the corresponding label file if it exists
            source_base = os.path.splitext(os.path.basename(source_path))[0]
            potential_label_paths = [
                os.path.join(os.path.dirname(os.path.dirname(source_path)), 'labels', f"{source_base}.txt"),
                os.path.join(os.path.dirname(source_path), f"{source_base}.txt")
            ]
            
            label_copied = False
            for label_path in potential_label_paths:
                if os.path.exists(label_path):
                    # Create a new label file with the same annotations
                    new_label_path = os.path.join(output_dir, 'labels', f"{os.path.splitext(synthetic_filename)[0]}.txt")
                    shutil.copy2(label_path, new_label_path)
                    results['generated_labels'].append(new_label_path)
                    label_copied = True
                    break
            
            if not label_copied:
                # If no label file found, create one with estimated annotation
                # This is a placeholder - in a real implementation, you would adjust
                # bounding box coordinates based on transformations applied
                new_label_path = os.path.join(output_dir, 'labels', f"{os.path.splitext(synthetic_filename)[0]}.txt")
                with open(new_label_path, 'w') as f:
                    # Format: class_id center_x center_y width height (normalized)
                    f.write(f"{class_id} 0.5 0.5 0.5 0.5\n")
                results['generated_labels'].append(new_label_path)
                print(f"Warning: Created estimated label for {synthetic_filename}")
            
            generated_count += 1
            
            # Progress update every 10 images
            if generated_count % 10 == 0:
                print(f"Generated {generated_count}/{count_needed} synthetic images")
                
        except Exception as e:
            print(f"Error saving synthetic image {output_path}: {e}")
    
    # Update results with final stats
    results['count_generated'] = generated_count
    results['techniques_used'] = list(techniques_used)
    
    print(f"Successfully generated {generated_count} synthetic images for {class_name}")
    print(f"Output directory: {output_dir}")
    print(f"Augmentation techniques used: {', '.join(techniques_used)}")
    
    return results


def visualize_class_distribution(df_bboxes, class_names=None, output_path=None):
    """
    Creates a bar chart showing the distribution of classes.
    
    Args:
        df_bboxes: DataFrame with bounding boxes containing 'class_id'
        class_names: Dict mapping class_id to human-readable names
        output_path: If provided, saves the chart to this path
        
    Returns:
        None (displays or saves the chart)
    """
    if 'class_id' not in df_bboxes.columns:
        raise ValueError("DataFrame must contain a 'class_id' column")
    
    # Count occurrences by class
    class_counts = df_bboxes['class_id'].value_counts().sort_index()
    
    # Setup the plot
    plt.figure(figsize=(12, 6))
    
    # Get x-labels based on class names if available
    if class_names:
        x_labels = [class_names.get(int(idx), f"Class {idx}") for idx in class_counts.index]
    else:
        x_labels = [f"Class {idx}" for idx in class_counts.index]
    
    # Create the bar chart
    bars = plt.bar(range(len(class_counts)), class_counts.values)
    
    # Add value labels on top of bars
    for i, v in enumerate(class_counts.values):
        plt.text(i, v + 5, str(v), ha='center')
    
    # Set labels and title
    plt.xlabel('Class')
    plt.ylabel('Number of Instances')
    plt.title('Distribution of Classes in Dataset')
    plt.xticks(range(len(class_counts)), x_labels, rotation=45, ha='right')
    
    # Add horizontal line for suggested minimum class size
    suggested_min = max(50, int(class_counts.median() * 0.5))
    plt.axhline(y=suggested_min, color='r', linestyle='--', 
                label=f'Suggested Minimum ({suggested_min})')
    
    plt.legend()
    plt.tight_layout()
    
    # Save or show
    if output_path:
        plt.savefig(output_path)
        print(f"Class distribution chart saved to {output_path}")
    else:
        plt.show()


def suggest_augmentation_strategy(df_bboxes, target_count=None):
    """
    Suggests augmentation strategies for underrepresented classes.
    
    Args:
        df_bboxes: DataFrame with bounding box information including 'class_id'
        target_count: Target count for each class (defaults to median class count)
        
    Returns:
        DataFrame with augmentation suggestions for each class
    """
    if 'class_id' not in df_bboxes.columns:
        raise ValueError("DataFrame must contain a 'class_id' column")
        
    # Count occurrences of each class
    class_counts = df_bboxes['class_id'].value_counts()
    
    # Set target count as median if not specified
    if target_count is None:
        target_count = int(class_counts.median())
    
    # Create suggestions DataFrame
    suggestions = []
    
    for class_id, count in class_counts.items():
        if count < target_count:
            deficit = target_count - count
            multiplier = target_count / count
            
            # Determine augmentation strategy based on deficit
            if multiplier <= 2:
                strategy = "Basic augmentation (flips, rotations)"
                technique = "geometric"
            elif multiplier <= 5:
                strategy = "Advanced augmentation (color shifts, noise, blur)"
                technique = "appearance"
            else:
                strategy = "Generate synthetic data or collect more samples"
                technique = "synthetic"
                
            suggestions.append({
                'class_id': class_id,
                'current_count': count,
                'target_count': target_count,
                'deficit': deficit,
                'multiplier_needed': round(multiplier, 2),
                'suggested_strategy': strategy,
                'technique': technique
            })
    
    return pd.DataFrame(suggestions)


def balance_dataset_by_sampling(df_bboxes, strategy='undersample', target_count=None):
    """
    Balances the dataset by sampling.
    
    Args:
        df_bboxes: DataFrame with bounding box information including 'class_id'
        strategy: 'undersample' to reduce overrepresented classes,
                 'oversample' to duplicate underrepresented classes
        target_count: Target count for each class
        
    Returns:
        Balanced DataFrame
    """
    if 'class_id' not in df_bboxes.columns:
        raise ValueError("DataFrame must contain a 'class_id' column")
        
    # Group by class_id and filename to keep boxes from same image together
    if 'filename' not in df_bboxes.columns:
        print("Warning: 'filename' column not found, treating each row independently")
        grouped = df_bboxes.groupby('class_id')
    else:
        # First group by class and filename to keep related boxes together
        grouped = df_bboxes.groupby(['class_id', 'filename'])
    
    # Count unique images per class
    if 'filename' in df_bboxes.columns:
        class_image_counts = df_bboxes.groupby('class_id')['filename'].nunique()
    else:
        class_image_counts = df_bboxes['class_id'].value_counts()
    
    # Set target count
    if target_count is None:
        if strategy == 'undersample':
            target_count = int(class_image_counts.min())
        else:  # oversample
            target_count = int(class_image_counts.max())
    
    # Initialize result
    balanced_df = pd.DataFrame()
    
    # Process each class
    for class_id, count in class_image_counts.items():
        class_groups = [group for name, group in grouped if name[0] == class_id] if 'filename' in df_bboxes.columns else \
                      [grouped.get_group(class_id)]
        
        if strategy == 'undersample' and count > target_count:
            # Randomly sample without replacement
            sampled_groups = random.sample(class_groups, target_count)
            class_df = pd.concat(sampled_groups)
            
        elif strategy == 'oversample' and count < target_count:
            # Sample with replacement to reach target
            needed = target_count - count
            extra_groups = random.choices(class_groups, k=needed)
            class_df = pd.concat(class_groups + extra_groups)
            
        else:
            # Keep as is
            class_df = pd.concat(class_groups)
        
        balanced_df = pd.concat([balanced_df, class_df])
    
    return balanced_df.reset_index(drop=True)


def report_class_balance_issues(df_bboxes, threshold=50, class_names=None):
    """
    Generates a comprehensive report on class balance issues.
    
    Args:
        df_bboxes: DataFrame with bounding box information
        threshold: Minimum samples per class
        class_names: Dict mapping class IDs to names
        
    Returns:
        Dict with balance information and issues
    """
    if 'class_id' not in df_bboxes.columns:
        raise ValueError("DataFrame must contain a 'class_id' column")
    
    # Count by class
    class_counts = df_bboxes['class_id'].value_counts().sort_index()
    
    # Find class imbalance issues
    total_boxes = len(df_bboxes)
    class_percentages = (class_counts / total_boxes * 100).round(2)
    
    # Get underrepresented classes
    underrepresented = find_underrepresented_classes(df_bboxes, threshold)
    
    # Find severely imbalanced classes (>20% of dataset)
    severely_imbalanced = class_percentages[class_percentages > 20].index.tolist()
    
    # Calculate imbalance metrics
    gini_coefficient = 1 - (1 / len(class_counts)) * (2 * sum((i+1) * count for i, count in enumerate(sorted(class_counts)))) / (total_boxes * len(class_counts))
    max_to_min_ratio = class_counts.max() / class_counts.min() if class_counts.min() > 0 else float('inf')
    
    # Compile the report
    report = {
        'total_boxes': total_boxes,
        'total_classes': len(class_counts),
        'class_counts': class_counts.to_dict(),
        'class_percentages': class_percentages.to_dict(),
        'underrepresented_classes': underrepresented,
        'severely_imbalanced_classes': severely_imbalanced,
        'gini_coefficient': round(gini_coefficient, 3),  # 0=perfectly balanced, 1=completely imbalanced
        'max_to_min_ratio': round(max_to_min_ratio, 1),
        'median_class_count': int(class_counts.median()),
        'is_balanced': len(underrepresented) == 0 and len(severely_imbalanced) == 0
    }
    
    # Add human-readable class names if provided
    if class_names:
        name_mapping = {}
        for class_id in class_counts.index:
            name_mapping[int(class_id)] = class_names.get(int(class_id), f"Class {class_id}")
        report['class_names'] = name_mapping
    
    # Generate recommendations
    recommendations = []
    
    if report['gini_coefficient'] > 0.3:
        recommendations.append("Dataset shows significant class imbalance")
        
    if len(underrepresented) > 0:
        classes_str = ", ".join([str(c) for c in underrepresented[:5]])
        if len(underrepresented) > 5:
            classes_str += f" and {len(underrepresented) - 5} more"
        recommendations.append(f"Collect more data for underrepresented classes: {classes_str}")
    
    if len(severely_imbalanced) > 0:
        classes_str = ", ".join([str(c) for c in severely_imbalanced])
        recommendations.append(f"Consider downsampling overrepresented classes: {classes_str}")
        
    report['recommendations'] = recommendations
    
    return report


if __name__ == "__main__":
    # Example usage
    print("Example usage of underrepresented_classes module:")
    
    # Create a sample DataFrame with unbalanced classes
    sample_data = []
    for i in range(5):  # 5 classes
        # Create varying numbers of samples per class
        count = 30 if i == 2 else 100 + i * 50
        for j in range(count):
            sample_data.append({
                'filename': f"img_{i}_{j//3}.jpg",  # Group boxes into images
                'class_id': i,
                'xmin': 100,
                'ymin': 100,
                'xmax': 200,
                'ymax': 200,
                'width': 100,
                'height': 100
            })
    
    df = pd.DataFrame(sample_data)
    print(f"Created sample DataFrame with {len(df)} bounding boxes")
    
    # Find underrepresented classes
    under = find_underrepresented_classes(df, threshold=50)
    print(f"Underrepresented classes: {under}")
    
    # Get suggestions for augmentation
    suggestions = suggest_augmentation_strategy(df)
    if not suggestions.empty:
        print("\nAugmentation suggestions:")
        print(suggestions)
    
    # Generate report
    report = report_class_balance_issues(df)
    print("\nClass balance report:")
    for key, value in report.items():
        if not isinstance(value, dict):
            print(f"{key}: {value}")
    
    # Visualize
    print("\nGenerating class distribution visualization...")
    visualize_class_distribution(df)
