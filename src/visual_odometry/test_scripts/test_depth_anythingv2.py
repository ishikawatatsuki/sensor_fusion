import torch
import cv2
import numpy as np
from src.visual_odometry.models.da_v2.depth_anything_v2.dpt import DepthAnythingV2

model_configs = {
    'vits': {'encoder': 'vits', 'features': 64, 'out_channels': [48, 96, 192, 384]},
    'vitb': {'encoder': 'vitb', 'features': 128, 'out_channels': [96, 192, 384, 768]},
    'vitl': {'encoder': 'vitl', 'features': 256, 'out_channels': [256, 512, 1024, 1024]}
}

encoder = 'vitb' # or 'vits', 'vitb'
dataset = 'hypersim' # 'hypersim' for indoor model, 'vkitti' for outdoor model
max_depth = 20 # 20 for indoor model, 80 for outdoor model

model = DepthAnythingV2(**{**model_configs[encoder]})
device = torch.device('mps' if torch.mps.is_available() else 'cpu')

model.load_state_dict(torch.load(f'/Volumes/Data_EXT/data/workspaces/sensor_fusion/src/visual_odometry/depth_anything_weights/depth_anything_v2_metric_hypersim_vitb.pth', map_location=device))
model = model.to(device)
model.eval()

base_dir = "/Volumes/Data_EXT/data/workspaces/sensor_fusion/data/KITTI/2011_10_03/2011_10_03_drive_0042_sync/image_02/data/0000000222.png"
raw_img = cv2.imread(base_dir)

# infer_image expects a numpy array (raw BGR image), not a tensor
depth = model.infer_image(raw_img) # HxW depth map in meters in numpy

print(f"Depth shape: {depth.shape}")
print(f"Depth range: min={depth.min():.4f}, max={depth.max():.4f}, mean={depth.mean():.4f}")

# Save raw depth as .npy for use in application (preserves metric values)
np.save('src/visual_odometry/test_scripts/depth_output.npy', depth)

# For visualization only - create normalized colored version
depth_normalized = (depth - depth.min()) / (depth.max() - depth.min() + 1e-8)
depth_vis = (depth_normalized * 255).astype(np.uint8)
depth_colored = cv2.applyColorMap(depth_vis, cv2.COLORMAP_INFERNO)

cv2.imwrite('src/visual_odometry/test_scripts/depth_output_colored.png', depth_colored)
print("Depth map saved! Raw metric depth: depth_output.npy, Visualization: depth_output_colored.png")