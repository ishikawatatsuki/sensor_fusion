import os
import cv2
import torch
import logging
import numpy as np
from PIL import Image
from enum import Enum
from ultralytics import YOLO
from transformers import pipeline
from einops import rearrange

class ObjectDetectorType(Enum):
    YOLO = "yolo"
    SEGFORMER = "segformer"
    @classmethod
    def from_string(cls, value: str):
        if value.lower() == 'yolo':
            return cls.YOLO
        elif value.lower() == 'segformer':
            return cls.SEGFORMER
        else:
            logging.warning(f"Unknown object detector type: {value}, defaulting to YOLO")
            return cls.YOLO
        
    @staticmethod
    def get_model_path(model_type_str: str) -> str:
        model_type_str = model_type_str.lower()
        
        match (ObjectDetectorType.from_string(model_type_str)):
            case ObjectDetectorType.YOLO:
                if os.path.exists("src/visual_odometry/yolo11n-seg.pt"):
                    return "src/visual_odometry/yolo11n-seg.pt"
                logging.warning("yolov11 segmentation model path does not exist.")
                return "yolov11n-seg.pt"
            
            case ObjectDetectorType.SEGFORMER:
                if os.path.exists("src/visual_odometry/segformer_model"):
                    return "src/visual_odometry/segformer_model"
                logging.warning("Segformer model path does not exist.")
                return "nvidia/segformer-b1-finetuned-cityscapes-1024-1024"
            
            case _:
                return "yolov11n-seg.pt"

class DynamicObjectDetector:
    def __init__(
            self, 
            type=ObjectDetectorType,
            model_path="yolo11n-seg.pt",
            dynamic_classes=None, 
            conf=0.6
            ):

        device = "cuda" if torch.cuda.is_available() else "cpu"
        if torch.mps.is_available():
            device = "mps"
        
        self.tye = type
        self.device = torch.device(device)
        
        if type == ObjectDetectorType.YOLO:
            self.model = YOLO(model_path).to(self.device)
            self.runner = self._run_yolo
        else:
            self.model = pipeline("image-segmentation", model_path, use_fast=True, device=device)
            if not os.path.exists("src/visual_odometry/segformer_model"):
                model_path = "src/visual_odometry/segformer_model"
                self.model.model.save_pretrained(model_path)
            self.runner = self._run_segformer

        self.conf = conf
        if dynamic_classes is None:
            dynamic_classes = ["person","car","truck","bus","motorbike","bicycle", "sky"]
        self.dynamic_classes = set(dynamic_classes)

    def _run_yolo(self, image: np.ndarray) -> np.ndarray:
        if len(image.shape) == 2:
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        H, W = image.shape[:2]
        mask = np.zeros((H, W), dtype=np.uint8)

        results = self.model.predict(source=image, conf=self.conf)[0]
        segments = results.masks.data.cpu().numpy() if results.masks is not None else []
        boxes = results.boxes.data.cpu().numpy() if results.boxes is not None else []

        logging.debug(f"Detected {len(segments)} segments and {len(boxes)} boxes.")
        for seg, cls in zip(segments, boxes):
            label = self.model.names[int(np.argmax(cls))]
            if label in self.dynamic_classes:
                seg_np = np.array(seg).astype(np.uint8) * 255  # (224, 640)
                resized = cv2.resize(seg_np, (W, H), interpolation=cv2.INTER_NEAREST)  # match image size
                mask = np.maximum(mask, resized)  # combine masks (or use |= if binary)

        return (mask == 0).astype(np.uint8) * 255
    
    def _run_segformer(self, image: np.ndarray) -> np.ndarray:
        if len(image.shape) == 2:
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        H, W = image.shape[:2]
        pil_image = Image.fromarray(image)
        outputs = self.model(pil_image)
        combined_mask = np.zeros((H, W), dtype=np.uint8)

        for pred in outputs:
            label = pred['label']
            if label in self.dynamic_classes:
                mask_np = np.array(pred['mask'], dtype=np.uint8)
                combined_mask = np.maximum(combined_mask, mask_np)
        
        return (combined_mask == 0).astype(np.uint8) * 255

    def _return_valid_mask(self, mask: np.ndarray, threshold_ratio: float = 0.5) -> np.ndarray:
        """
        Check if mask contains more than threshold_ratio of zeros.
        If so, return a mask filled with 255 (all valid).
        
        Args:
            mask (np.ndarray): Input binary mask with values {0, 255}.
            threshold_ratio (float): Ratio of zeros to total pixels that is considered suspicious.
        
        Returns:
            np.ndarray: Validated mask.
        """
        total_pixels = mask.size
        zero_pixels = np.count_nonzero(mask == 0)

        if zero_pixels > threshold_ratio * total_pixels:
            # suspicious mask, override with all 255
            logging.warning("Suspicious mask detected, overriding with all valid mask.")
            return np.full(mask.shape, 255, dtype=np.uint8)
        return mask

    def get_dynamic_mask(self, image: np.ndarray) -> np.ndarray:
        mask = self.runner(image)
        return self._return_valid_mask(mask)
    
if __name__ == "__main__":
    import cv2
    import os
    import numpy as np
    import requests
    import time
    import matplotlib.pyplot as plt
    from .vo_utils import (
        filter_keypoints_by_border,
        filter_keypoints_by_response,
        filter_keypoints_by_size
    )

    # Example usage
    type_obj = "segformer"
    detector_type = ObjectDetectorType.from_string(type_obj)
    model_path = ObjectDetectorType.get_model_path(type_obj)
    print(f"Using model path: {model_path}, detector type: {detector_type}")
    masker = DynamicObjectDetector(
        type=detector_type,
        model_path=model_path,
        dynamic_classes=["person", "car", "bicycle", "motorbike", "bus", "truck", "sky"],
        conf=0.6)

    # url = "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/transformers/tasks/segmentation_input.jpg"
    # image = Image.open(requests.get(url, stream=True).raw)
    # image = np.array(image)

    # base_dir = "/Volumes/Data_EXT/data/workspaces/sensor_fusion/data/KITTI/2011_10_03/2011_10_03_drive_0042_sync/image_00/data/0000000222.png"
    base_dir = "/Volumes/Data_EXT/data/workspaces/sensor_fusion/data/EuRoC/MH_01_easy/cam0/data/1403636588313555456.png"
    image = cv2.imread(base_dir)
    print(f"Image shape: {image.shape}")
    start = time.time()
    mask = masker.get_dynamic_mask(image)
    print(mask)
    end = time.time()

    print(f"Detection time: {end-start}")

    # mask = np.zeros(mask.shape, dtype=np.uint8)  # Reset mask for testing
    # mask[100:200, 100:200] = 255  # Example mask region

    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray_image = rearrange(gray_image, 'h w -> h w 1')  # Ensure it has 3 channels for visualization
    print(f"gray image shape: {gray_image.shape}")
    points = cv2.goodFeaturesToTrack(gray_image, maxCorners=1000, qualityLevel=0.01, minDistance=7, mask=mask)

    detector = cv2.SIFT().create(nfeatures=1000, sigma=1.6)
    keypoints, descriptors = detector.detectAndCompute(image, mask=mask)
    
    # keypoints, descriptors = filter_keypoints_by_border(keypoints, descriptors, image.shape, border=30)
    # keypoints, descriptors = filter_keypoints_by_response(keypoints, descriptors, threshold=0.01)
    # keypoints, descriptors = filter_keypoints_by_size(keypoints, descriptors, min_size=2.0, max_size=10.0)

    show_image = True
    if show_image:
        # Display the mask
        cv2.namedWindow("Dynamic Mask", cv2.WINDOW_NORMAL)
        # cv2.imshow("Dynamic Mask", mask)
        plt.imshow(mask)
        plt.title("Dynamic Object Mask")
        plt.axis('off')
        plt.show()
        if points is not None:
            for point in points:
                x, y = point.ravel()
                cv2.circle(image, (int(x), int(y)), 3, (0, 255, 0), -1)
        if keypoints:
            image_with_keypoints = cv2.drawKeypoints(image, keypoints, None, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
            cv2.imshow("Keypoints (SIFT)", image_with_keypoints)
        cv2.imshow("Dynamic Mask with Features", image)

        while True:
            if cv2.waitKey(10) & 0xFF == ord('q'):  # Press 'q' to quit
                break

        cv2.destroyAllWindows()
    else:
        out_dir = "src/visual_odometry/output_visualization_test"
        os.makedirs(out_dir, exist_ok=True)

        cv2.imwrite(os.path.join(out_dir, "dynamic_mask.png"), mask)
        cv2.imwrite(os.path.join(out_dir, "gray_with_corners.png"), gray_image)
        cv2.imwrite(os.path.join(out_dir, "final_features.png"), image)
        if keypoints:
            image_with_keypoints = cv2.drawKeypoints(image, keypoints, None, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
            cv2.imwrite(os.path.join(out_dir, "sift_keypoints.png"), image_with_keypoints)
        

        print("Saved all visualization outputs to:", out_dir)