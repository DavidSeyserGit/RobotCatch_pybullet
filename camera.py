import pybullet as p
import pybullet_data
import numpy as np
import cv2
import math

class RSD435:
    def __init__(self):
        self.nearPlane = 0.28
        self.farPlane = 3
        self.resolutionDepth = (1280, 720)
        self.resolutionRGB = (1920, 1080)
        self.fovDepth = (87, 58) # Horizontal, Vertical FOV
        self.fovRGB = (69, 42) # Horizontal, Vertical FOV
        self.cameraPosition = [-2.5, 0, 1]
        self.viewMatrix = p.computeViewMatrix(self.cameraPosition, [1, 0, 0], [0, 0, 1]) #cameraTargetPosition
        self.projectionMatrixDepth = p.computeProjectionMatrixFOV(self.fovDepth[1], self.resolutionDepth[0] / self.resolutionDepth[1], self.nearPlane, self.farPlane)
        self.projectionMatrixRGB = p.computeProjectionMatrixFOV(self.fovRGB[1], self.resolutionRGB[0] / self.resolutionRGB[1], self.nearPlane, self.farPlane)
    
    def getDepthImage(self):
        _, _, _, depth_img, _ = p.getCameraImage(
        camera.resolutionDepth[0], camera.resolutionDepth[1],
        camera.viewMatrix, camera.projectionMatrixDepth
        )
        depth_img = np.reshape(depth_img, (camera.resolutionDepth[1], camera.resolutionDepth[0]))
        depth_normalized = (depth_img - np.min(depth_img)) / (np.max(depth_img) - np.min(depth_img)) * 255
        depth_normalized = depth_normalized.astype(np.uint8)  # Convert to unsigned 8-bit integer
        return depth_normalized
    
    def getRGBImage(self):
        _, _, rgb_image, _, _ = p.getCameraImage(
        camera.resolutionRGB[0], camera.resolutionRGB[1],
        camera.viewMatrix, camera.projectionMatrixRGB
        )
        rgb_image = np.reshape(rgb_image, (camera.resolutionRGB[1], camera.resolutionRGB[0], 4))  # Reshape with 4 channels (RGBA)
        rgb_image = rgb_image[:, :, :3]  # Discard the alpha channel
        return rgb_image


    def create_camera_box(self, position, size=0.05):
        visual_shape_id = p.createVisualShape(shapeType=p.GEOM_BOX, halfExtents=[size/2]*3, rgbaColor=[1, 0, 0, 1])
        collison_shape_id = p.createCollisionShape(shapeType=p.GEOM_BOX, halfExtents=[size/2]*3)
        p.createMultiBody(baseMass=0, baseVisualShapeIndex=visual_shape_id, baseCollisionShapeIndex=collison_shape_id, basePosition=position)

    def create_fov_lines(self, camera_position, fov_horizontal, fov_vertical, line_length=1):
        fov_h_rad = math.radians(fov_horizontal / 2)
        fov_v_rad = math.radians(fov_vertical / 2)
        left_h = [
            camera_position[0] + line_length * math.cos(fov_h_rad),
            camera_position[1] - line_length * math.sin(fov_h_rad),
            camera_position[2]
        ]
        right_h = [
            camera_position[0] + line_length * math.cos(fov_h_rad),
            camera_position[1] + line_length * math.sin(fov_h_rad),
            camera_position[2]
        ]
        top_v = [
            camera_position[0] + line_length,
            camera_position[1],
            camera_position[2] + line_length * math.tan(fov_v_rad)
        ]
        bottom_v = [
            camera_position[0] + line_length,
            camera_position[1],
            camera_position[2] - line_length * math.tan(fov_v_rad)
        ]

        p.addUserDebugLine(camera_position, left_h, [0, 1, 0])
        p.addUserDebugLine(camera_position, right_h, [0, 1, 0])
        p.addUserDebugLine(camera_position, top_v, [0, 0, 1])
        p.addUserDebugLine(camera_position, bottom_v, [0, 0, 1])



if __name__ == "__main__":
    p.connect(p.GUI)
    p.setAdditionalSearchPath(pybullet_data.getDataPath())
    p.loadURDF("plane.urdf")
    p.loadURDF("r2d2.urdf", [0, 0, 0.5])
    p.setGravity(0, 0, -9.8)

    camera = RSD435()

    camera.create_camera_box(camera.cameraPosition)
    camera.create_fov_lines(camera.cameraPosition, camera.fovDepth[0], camera.fovDepth[1])

    depth_img = camera.getDepthImage() 
    rgb_image = camera.getRGBImage()

    # Display the depth image using OpenCV
    cv2.imshow("Depth Image", depth_img)
    cv2.imshow("RGB Image", rgb_image)
    cv2.waitKey(0)  # Press any key to close the window
    cv2.destroyAllWindows()

    # Keep the simulation running
    while p.isConnected():
        p.stepSimulation()

    # Disconnect from PyBullet
    p.disconnect()
