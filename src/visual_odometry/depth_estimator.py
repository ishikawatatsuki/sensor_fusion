import cv2
import logging
import torch
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from transformers import AutoImageProcessor, AutoModelForDepthEstimation, DPTImageProcessor, DPTForDepthEstimation
from einops import rearrange
from enum import Enum

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.common.config import VisualOdometryConfig

class DepthEstimatorType(Enum):
    ZoeMap = "zoe_depth"
    DinoV2 = "dino_v2"

    DepthAnything = 'depth_anything'
    Midas = "dpt_midas"

    @classmethod
    def from_string(cls, value: str):
        if value.lower() == 'depth_anything':
            return cls.DepthAnything
        elif value.lower() == 'zoe_depth':
            return cls.ZoeMap
        elif value.lower() == 'dpt_midas':
            return cls.Midas
        elif value.lower() == 'dino_v2':
            return cls.DinoV2
        else:
            raise ValueError(f"Unknown depth estimator type: {value}")


class DepthEstimator:

    def __init__(self, config: VisualOdometryConfig):
        self.config = config
        self.depth_estimator = DepthEstimatorType.from_string(config.depth_estimator)
        self.device = self._get_device()
        
        self.image_processor = self._get_image_processor()
        self.model = self._get_model().to(self.device).eval()

    def _get_device(self):
        if torch.cuda.is_available():
            return torch.device("cuda")
        elif torch.mps.is_available():
            return torch.device("mps")
        else:
            return torch.device("cpu")


    def _get_image_processor(self):
        if self.depth_estimator == DepthEstimatorType.DepthAnything:
            return AutoImageProcessor.from_pretrained("depth-anything/Depth-Anything-V2-Small-hf")
        elif self.depth_estimator == DepthEstimatorType.ZoeMap:
            return AutoImageProcessor.from_pretrained("Intel/zoedepth-nyu-kitti")
        elif self.depth_estimator == DepthEstimatorType.Midas:
            return DPTImageProcessor.from_pretrained("Intel/dpt-hybrid-midas")
        elif self.depth_estimator == DepthEstimatorType.DinoV2:
            return AutoImageProcessor.from_pretrained("facebook/dpt-dinov2-base-kitti")
        else:
            raise ValueError(f"Unknown depth estimator type: {self.depth_estimator}")

    def _get_model(self):
        if self.depth_estimator == DepthEstimatorType.DepthAnything:
            return AutoModelForDepthEstimation.from_pretrained("depth-anything/Depth-Anything-V2-Small-hf")
        elif self.depth_estimator == DepthEstimatorType.ZoeMap:
            return AutoModelForDepthEstimation.from_pretrained("Intel/zoedepth-nyu-kitti")
        elif self.depth_estimator == DepthEstimatorType.Midas:
            return DPTForDepthEstimation.from_pretrained("Intel/dpt-hybrid-midas", low_cpu_mem_usage=True)
        elif self.depth_estimator == DepthEstimatorType.DinoV2:
            return AutoModelForDepthEstimation.from_pretrained("facebook/dpt-dinov2-base-kitti")
        else:
            raise ValueError(f"Unknown depth estimator type: {self.depth_estimator}")

    def estimate_depth(self, frame: np.ndarray) -> np.ndarray:
        image = Image.fromarray(frame).convert("L")
        rgb = Image.merge("RGB", (image, image, image))  # Convert to RGB

        logging.debug("Estimating depth...")
        logging.debug(f"Image shape: {frame.shape}")
            
        inputs = self.image_processor(images=rgb,
                                      return_tensors="pt").to(self.device)

        with torch.no_grad():
            outputs = self.model(**inputs)

        if self.depth_estimator == DepthEstimatorType.ZoeMap:
            post_processed_output = self.image_processor.post_process_depth_estimation(
                outputs,
                source_sizes=[(frame.shape[0], frame.shape[1])],
                target_sizes=[(frame.shape[0], frame.shape[1])],
            )
            depth = post_processed_output[0]["predicted_depth"]
            depth = depth.detach().cpu().numpy().astype("float32")
        elif self.depth_estimator == DepthEstimatorType.Midas:
            predicted_depth = outputs.predicted_depth
            prediction = torch.nn.functional.interpolate(
                predicted_depth.unsqueeze(1),
                size=image.size[::-1],
                mode="bicubic",
                align_corners=False,
            )
            outputs = prediction.squeeze().cpu().numpy()
            depth = ((np.max(outputs) - outputs) * 255 / np.max(outputs)).astype("float32")
        elif self.depth_estimator == DepthEstimatorType.DinoV2:
            predicted_depth = outputs.predicted_depth
            prediction = torch.nn.functional.interpolate(
                predicted_depth.unsqueeze(1),
                size=image.size[::-1],
                mode="bicubic",
                align_corners=False,
            )
            depth = prediction.squeeze().cpu().numpy().astype("float32")
        else:
            post_processed_output = self.image_processor.post_process_depth_estimation(
                outputs,
                target_sizes=[(frame.shape[0], frame.shape[1])],
            )
            outputs = post_processed_output[0]["predicted_depth"]
            depth = outputs.detach().cpu().numpy().astype("float32")
        
        logging.debug(f"Depth shape: {depth.shape}")  

        return depth

def visualize_depth_overlay(image, depth_map, stride=20):
    img_vis = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR) if len(image.shape) == 2 else image.copy()

    for y in range(0, image.shape[0], stride):
        for x in range(0, image.shape[1], stride):
            z = depth_map[y, x]
            if not np.isnan(z) and 0.1 < z < 80:
                cv2.putText(img_vis, f"{z:.1f}", (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0, 255, 0), 1)

    plt.imshow(img_vis[..., ::-1])
    plt.title("Depth Overlay")
    plt.axis('off')
    plt.show()



if __name__ == "__main__":
    import requests

    device = torch.device("mps" if torch.mps.is_available() else "cpu")
    # url = 'http://images.cocodataset.org/val2017/000000039769.jpg'
    # image = Image.open(requests.get(url, stream=True).raw)
    base_dir = "/Volumes/Data_EXT/data/workspaces/sensor_fusion/data/KITTI/2011_09_30/2011_09_30_drive_0020_sync/image_00/data/0000000006.png"
    image = cv2.imread(base_dir)
    print(image.shape)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    print(image.shape)
    image = Image.fromarray(image)

    config = VisualOdometryConfig(
        type='monocular',
        estimator='2d3d',
        camera_id='left',
        depth_estimator='depth_anything',
        params={
            'max_features': 1000,
            'ransac_reproj_threshold': 1.0,
            'confidence': 0.999,
            'min_inliers': 50
        }
    )

    depth_estimator = DepthEstimator(config=config)
    depth = depth_estimator.estimate_depth(np.array(image))
    depth_np = np.array(depth)
    print(depth_np.shape)
    print(depth_np.max(), depth_np.min())
    print(depth_np)
    depth = Image.fromarray(depth.astype("uint8"))
    depth.save("depth_estimation_output.png")

    # Visualize the depth map
    plt.imshow(depth, cmap="inferno")
    plt.colorbar()
    plt.title('Depth Estimation Output')
    plt.axis('off')
    plt.show()

    # visualize_depth_overlay(np.array(image), depth_np)
