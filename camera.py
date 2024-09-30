import pybullet as p
import pybullet_data
import numpy as np
import threading
import cv2
import math
import random
import time
from ball import Ball, Simulation

class RSD435:
    def __init__(self, cameraPosition=[0, 0, 0], targetPosition=[1, 0, 0]):
        self.nearPlane = 0.28
        self.farPlane = 3
        self.resolutionDepth = (640, 360)
        self.resolutionRGB = (960, 540)
        self.fovDepth = (87, 58)  # Horizontal, Vertical FOV
        self.fovRGB = (69, 42)  # Horizontal, Vertical FOV
        self.cameraPosition = cameraPosition
        self.targetPosition = targetPosition
        self.update_view_and_projection()

    def update_view_and_projection(self):
        """Update view and projection matrices based on current camera position and target position."""
        self.viewMatrix = p.computeViewMatrix(self.cameraPosition, self.targetPosition, [0, 0, 1])
        self.projectionMatrixDepth = p.computeProjectionMatrixFOV(
            self.fovDepth[1], self.resolutionDepth[0] / self.resolutionDepth[1], self.nearPlane, self.farPlane
        )
        self.projectionMatrixRGB = p.computeProjectionMatrixFOV(
            self.fovRGB[1], self.resolutionRGB[0] / self.resolutionRGB[1], self.nearPlane, self.farPlane
        )

    def getDepthImage(self):
        """Capture and return the depth image from the camera."""
        try:
            # 1. Use specific return values
            _, _, _, depth_img, _ = p.getCameraImage(
                width=self.resolutionDepth[0],
                height=self.resolutionDepth[1],
                viewMatrix=self.viewMatrix,
                projectionMatrix=self.projectionMatrixDepth,
                flags=p.ER_NO_SEGMENTATION_MASK
            )

            # 2. Reshape more efficiently
            depth_img = np.array(depth_img).reshape(self.resolutionDepth[1], self.resolutionDepth[0])

            # 3. Use cv2.normalize for more efficient normalization
            depth_normalized = cv2.normalize(depth_img, None, 0, 255, cv2.NORM_MINMAX)

            # 4. Convert to uint8 more efficiently
            return depth_normalized.astype(np.uint8)

        except Exception as e:
            print(f"Error capturing depth image: {e}")
            return None


    def getRGBImage(self):
        """Capture and return the RGB image from the camera."""
        try:
            _, _, rgb_image, _, _ = p.getCameraImage(
                self.resolutionRGB[0], self.resolutionRGB[1],
                self.viewMatrix, self.projectionMatrixRGB
            )
            rgb_image = np.reshape(rgb_image, (self.resolutionRGB[1], self.resolutionRGB[0], 4))  # Reshape with 4 channels (RGBA)
            rgb_image = rgb_image[:, :, :3]  # Discard the alpha channel
            return rgb_image
        except Exception as e:
            print(f"Error capturing RGB image: {e}")
            return None

    def create_camera_box(self, position, size=0.05):
        visual_shape_id = p.createVisualShape(shapeType=p.GEOM_BOX, halfExtents=[size/2]*3, rgbaColor=[1, 0, 0, 1])
        collision_shape_id = p.createCollisionShape(shapeType=p.GEOM_BOX, halfExtents=[size/2]*3)
        p.createMultiBody(baseMass=0, baseVisualShapeIndex=visual_shape_id, baseCollisionShapeIndex=collision_shape_id, basePosition=position)

def run_simulation(camera):
    """Function to run the simulation."""
    p.setRealTimeSimulation(0)  # Disable real-time simulation
    step_counter = 0

    while p.isConnected():
        p.stepSimulation()
        if step_counter % 480 == 0:
            ball.spawn()  # Spawn the ball or perform simulation logic
        step_counter += 1
        time.sleep(1.0 / 240)  # Sleep for a fixed time step (240 Hz)

def capture_images(camera):
    """Function to capture images."""
    last_capture_time = time.time()
    capture_interval = 0.05 # Minimum time interval between captures in seconds

    while p.isConnected():
        current_time = time.time()

        # Capture images based on time interval
        if current_time - last_capture_time >= capture_interval:
            depth_img = camera.getDepthImage()
            rgb_image = camera.getRGBImage()

            if depth_img is not None:
                cv2.imshow("Depth Image", depth_img)
            if rgb_image is not None:
                cv2.imshow("RGB Image", rgb_image)

            last_capture_time = current_time

            # Check for a key press to close the windows
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

if __name__ == "__main__":
    p.connect(p.DIRECT)
    p.setAdditionalSearchPath(pybullet_data.getDataPath())
    p.loadURDF("plane.urdf")
    
    r2d2_id = p.loadURDF("r2d2.urdf", [2, 1, 1])
    p.setGravity(0, 0, -9.8)

    z_velocity = random.uniform(1, 2)  # Random z velocity
    y_velocity = random.uniform(-0.5, 0.5)  # Random y velocity
    x_velocity = random.uniform(-8, -4)  # Random x velocity
    ball = Ball((3, 0, 1), (x_velocity, y_velocity, z_velocity))
    
    camera = RSD435([0, 0, 1], [1, 0.5, 1])

    camera.create_camera_box(camera.cameraPosition)
    #camera.create_fov_lines(camera.cameraPosition, camera.targetPosition, camera.fovRGB[0], camera.fovRGB[1])

    # Create threads for simulation and image capture
    simulation_thread = threading.Thread(target=run_simulation, args=(camera,))
    image_capture_thread = threading.Thread(target=capture_images, args=(camera,))

    # Start the threads
    simulation_thread.start()
    image_capture_thread.start()

    # Wait for the threads to finish
    simulation_thread.join()
    image_capture_thread.join()

    # Clean up
    cv2.destroyAllWindows()
    p.disconnect()