import cv2
import numpy as np
import matplotlib.pyplot as plt


def main(frame1, frame2):
    # Load two consecutive grayscale images
    img1 = cv2.imread(frame1, cv2.IMREAD_GRAYSCALE)
    img2 = cv2.imread(frame2, cv2.IMREAD_GRAYSCALE)

    # Compute dense optical flow using Farneback method
    flow = cv2.calcOpticalFlowFarneback(
        prev=img1,
        next=img2,
        flow=None,
        pyr_scale=0.5,
        levels=3,
        winsize=15,
        iterations=3,
        poly_n=5,
        poly_sigma=1.2,
        flags=0
    )

    # Compute magnitude and angle of flow
    magnitude, angle = cv2.cartToPolar(flow[..., 0], flow[..., 1])

    # Create HSV image to visualize flow
    hsv = np.zeros((img1.shape[0], img1.shape[1], 3), dtype=np.uint8)
    hsv[..., 0] = angle * 180 / np.pi / 2  # Hue: direction
    hsv[..., 1] = 255                      # Saturation
    hsv[..., 2] = cv2.normalize(magnitude, None, 0, 255, cv2.NORM_MINMAX)  # Value: magnitude

    # Convert HSV to RGB for displaying
    flow_rgb = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

    # Show the result
    plt.imshow(flow_rgb)
    plt.title("Dense Optical Flow (Farneback)")
    plt.axis("off")
    plt.show()


def main_lk(frame1, frame2):
    img1 = cv2.imread(frame1, cv2.IMREAD_GRAYSCALE)
    img2 = cv2.imread(frame2, cv2.IMREAD_GRAYSCALE)

    # Detect good features to track in the first image using Shi-Tomasi method
    feature_params = dict(maxCorners=200, qualityLevel=0.3, minDistance=7, blockSize=7)
    p0 = cv2.goodFeaturesToTrack(img1, mask=None, **feature_params)

    # Define LK optical flow parameters
    lk_params = dict(winSize=(15, 15),
                    maxLevel=2,
                    criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))

    # Calculate optical flow (track feature points)
    p1, st, err = cv2.calcOpticalFlowPyrLK(img1, img2, p0, None, **lk_params)

    # Select good points
    good_new = p1[st == 1]
    good_old = p0[st == 1]

    # Create an output image to draw on
    img2_color = cv2.cvtColor(img2, cv2.COLOR_GRAY2BGR)

    # Draw the tracks
    for i, (new, old) in enumerate(zip(good_new, good_old)):
        a, b = new.ravel()
        c, d = old.ravel()
        cv2.line(img2_color, (int(a), int(b)), (int(c), int(d)), (0, 255, 0), 2)
        cv2.circle(img2_color, (int(a), int(b)), 3, (0, 0, 255), -1)

    # Show the image with tracks
    cv2.imshow("Sparse Optical Flow - calcOpticalFlowPyrLK", img2_color)
    while True:
        if cv2.waitKey(10) & 0xFF == ord('q'):
            break
    cv2.destroyAllWindows()


if __name__ == "__main__":
    
    frame1 = "/Volumes/Data_EXT/data/workspaces/sensor_fusion/data/KITTI/2011_09_30/2011_09_30_drive_0020_sync/image_00/data/0000000007.png"
    frame2 = "/Volumes/Data_EXT/data/workspaces/sensor_fusion/data/KITTI/2011_09_30/2011_09_30_drive_0020_sync/image_00/data/0000000008.png"
    
    # main(frame1, frame2)
    main_lk(frame1, frame2)
