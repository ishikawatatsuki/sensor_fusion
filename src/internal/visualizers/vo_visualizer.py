import os
import cv2
import numpy as np
from datetime import datetime
import matplotlib.pyplot as plt
from multiprocessing import Process, Queue
from collections import deque
from typing import Deque
from dataclasses import dataclass

@dataclass
class VO_VisualizationData:
    frame: np.ndarray
    mask: np.ndarray
    pts_prev: np.ndarray
    pts_curr: np.ndarray
    estimated_pose: np.ndarray
    gt_pose: np.ndarray


class VO_Visualizer:
    def __init__(
            self, 
            save_path: str, 
            save_frame: bool = False,
            max_len=2000):
        self.queue = Queue()
        self.process = Process(target=self._run, args=(self.queue, max_len))
        self.process.daemon = True

        self.save_frame = save_frame
        self.save_folder = save_path
        if os.path.exists(self.save_folder):
            date_str = datetime.now().strftime("%Y%m%d_%H%M%S")
            self.save_folder = f"{self.save_folder}_{date_str}"
        
        if not os.path.exists(self.save_folder):
            os.makedirs(self.save_folder)

    def start(self):
        self.process.start()

    def stop(self):
        self.queue.put("EXIT")
        self.process.join()

    def send(self, data: VO_VisualizationData):
        self.queue.put(data)

    def _run(self, queue: Queue, max_len: int):
        estimated_trajectory: Deque[np.ndarray] = deque(maxlen=max_len)
        gt_trajectory: Deque[np.ndarray] = deque(maxlen=max_len)

        plt.ion()
        fig, axs = plt.subplots(2, 2, figsize=(10, 8))
        axs[1][1].axis("off")

        while True:
            data = queue.get()
            if data == "EXIT":
                break

            assert isinstance(data, VO_VisualizationData)

            frame = data.frame
            mask = data.mask
            pts_prev = data.pts_prev
            pts_curr = data.pts_curr
            estimated_pose = data.estimated_pose
            gt_pose = data.gt_pose
            estimated_trajectory.append(estimated_pose)
            gt_trajectory.append(gt_pose)

            # Convert BGR to RGB
            original_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # Draw tracked features
            tracked = frame.copy()
            if pts_prev is not None and pts_curr is not None:
                for p1, p2 in zip(pts_prev, pts_curr):
                    x1, y1 = map(int, p1)
                    x2, y2 = map(int, p2)
                    cv2.line(tracked, (x1, y1), (x2, y2), (255, 0, 0), 1)
                    cv2.circle(tracked, (x2, y2), 3, (0, 255, 0), -1)
            tracked_rgb = cv2.cvtColor(tracked, cv2.COLOR_BGR2RGB)

            # Clear and update all plots
            axs[0][0].clear()
            axs[1][0].clear()
            axs[0][1].clear()
            axs[1][1].clear()

            axs[0][0].imshow(original_rgb)
            axs[0][0].set_title("Original Frame")
            axs[0][0].axis("off")

            axs[1][1].imshow(mask)
            axs[1][1].set_title("Mask")
            axs[1][1].axis("off")
            
            axs[1][0].imshow(tracked_rgb)
            axs[1][0].set_title("Tracked Features")
            axs[1][0].axis("off")


            x_vals_est = [p[0] for p in estimated_trajectory]
            z_vals_est = [p[1] for p in estimated_trajectory]
            axs[0][1].plot(x_vals_est, z_vals_est, color='blue', label="Estimated Trajectory", linewidth=1)
            x_vals_gt = [p[0] for p in gt_trajectory]
            z_vals_gt = [p[1] for p in gt_trajectory]
            axs[0][1].plot(x_vals_gt, z_vals_gt, color='black', label="Ground Truth Trajectory", linewidth=1)
            axs[0][1].legend()
            axs[0][1].set_title("Estimated Trajectory (X-Z)")
            axs[0][1].set_xlabel("X[m]")
            axs[0][1].set_ylabel("Z[m]")
            axs[0][1].grid(True)

            fig.tight_layout()
            plt.pause(0.001)

            if self.save_frame:
                fig.savefig(os.path.join(self.save_folder, f"{len(estimated_trajectory)}.png"))


        plt.close()

if __name__ == "__main__":
    import numpy as np
    import time

    vis = VO_Visualizer(
        save_path="/Volumes/Data_EXT/data/workspaces/sensor_fusion/outputs/vo_estimates/vo_debug/07",
        save_frame=True
    )
    vis.start()

    for i in range(1000):
        
        img = np.full((480, 640, 3), 255, dtype=np.uint8)
        mask = np.full((480, 640, 3), 255, dtype=np.uint8)

        pts_prev = np.array([[100 + i, 100 + i], [200 + i, 200 + i]], dtype=np.float32)
        pts_curr = pts_prev + 5
        est_pose = np.array([i * 0.1, i * 0.1])  # x, z
        gt_pose = np.array([i * 0.1, i * 0.3])  # x, z

        vis_data = VO_VisualizationData(
            frame=img, 
            mask=mask,
            pts_prev=pts_prev, 
            pts_curr=pts_curr, 
            estimated_pose=est_pose, 
            gt_pose=gt_pose)
        vis.send(vis_data)
        time.sleep(0.05)

    vis.stop()