import cv2
from einops import rearrange
import torch
import logging
import numpy as np
from ultralytics import YOLO


class DynamicObjectDetector:
    def __init__(self, model_path="yolo11n-seg.pt", dynamic_classes=None, conf=0.3):

        device = "cuda" if torch.cuda.is_available() else "cpu"
        if torch.mps.is_available():
            device = "mps"
        
        self.device = torch.device(device)
        self.model = YOLO(model_path).to(self.device)
        self.conf = conf
        self.dynamic_classes = set(dynamic_classes or ["person","car","truck","bus","motorbike","bicycle"])

    def get_dynamic_mask(self, image: np.ndarray) -> np.ndarray:
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
    
if __name__ == "__main__":
    import cv2
    import numpy as np

    # Example usage
    masker = DynamicObjectDetector(
        model_path="yolo11n-seg.pt", 
        dynamic_classes=["person", "car", "bicycle", "motorbike", "bus", "truck"], 
        conf=0.6)

    base_dir = "/Volumes/Data_EXT/data/workspaces/sensor_fusion/data/KITTI/2011_09_30/2011_09_30_drive_0027_sync/image_00/data/0000000200.png"
    image = cv2.imread(base_dir)
    print(f"Image shape: {image.shape}")
    mask = masker.get_dynamic_mask(image)
    print(mask)
    mask = np.zeros(mask.shape, dtype=np.uint8)  # Reset mask for testing
    mask[100:200, 100:200] = 255  # Example mask region

    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray_image = rearrange(gray_image, 'h w -> h w 1')  # Ensure it has 3 channels for visualization
    print(f"gray image shape: {gray_image.shape}")
    points = cv2.goodFeaturesToTrack(gray_image, maxCorners=1000, qualityLevel=0.01, minDistance=7, mask=mask)

    # Display the mask
    cv2.namedWindow("Dynamic Mask", cv2.WINDOW_NORMAL)
    cv2.imshow("Dynamic Mask", mask)
    if points is not None:
        for point in points:
            x, y = point.ravel()
            cv2.circle(image, (int(x), int(y)), 3, (0, 255, 0), -1)

    detector = cv2.SIFT().create(nfeatures=1000, sigma=1.6)
    keypoints, descriptors = detector.detectAndCompute(image, mask=mask)
    if keypoints:
        image_with_keypoints = cv2.drawKeypoints(image, keypoints, None, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
        cv2.imshow("Keypoints (SIFT)", image_with_keypoints)

    cv2.imshow("Dynamic Mask with Features", image)

    while True:
        if cv2.waitKey(10) & 0xFF == ord('q'):  # Press 'q' to quit
            break

    cv2.destroyAllWindows()