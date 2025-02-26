"""YOLO model loading and integration utilities."""


def load_yolo_model(model_path, conf_thresh=0.4, iou_thresh=0.5):
    """
    Loads a YOLO model with specified thresholds.
    
    Args:
        model_path: Path to the YOLO .pt model file
        conf_thresh: Confidence threshold for detections
        iou_thresh: IoU threshold for non-maximum suppression
        
    Returns:
        Loaded YOLO model ready for inference
    """
    try:
        # Import here to avoid dependency if not needed
        from ultralytics import YOLO
        
        # Load the model
        model = YOLO(model_path)
        
        # Set the inference parameters
        model.conf = conf_thresh
        model.iou = iou_thresh
        
        print(f"Loaded YOLO model from {model_path}")
        print(f"Confidence threshold: {conf_thresh}")
        print(f"IoU threshold: {iou_thresh}")
        
        return model
    
    except ImportError:
        print("Error: ultralytics package not found.")
        print("Please install it: pip install ultralytics")
        return None
    except Exception as e:
        print(f"Error loading YOLO model: {e}")
        return None
