import pybullet as p
import pybullet_data
import numpy as np
import cv2
import math
import random

# Initialize PyBullet in DIRECT mode (software rendering)
physicsClient = p.connect(p.DIRECT)

p.setAdditionalSearchPath(pybullet_data.getDataPath())
p.setGravity(0, 0, -9.81)

# Load a plane
planeId = p.loadURDF("plane.urdf")

# Function to create a random color
def random_color():
    return [random.random(), random.random(), random.random(), 1]

# Add various objects to the scene
objects = []

# Add some boxes
for i in range(5):
    size = [random.uniform(0.05, 0.2)] * 3
    pos = [random.uniform(-1, 1), random.uniform(-1, 1), size[2]/2 + 1]  # Elevated initial position
    color = random_color()
    visual = p.createVisualShape(shapeType=p.GEOM_BOX, halfExtents=size, rgbaColor=color)
    collision = p.createCollisionShape(shapeType=p.GEOM_BOX, halfExtents=size)
    obj = p.createMultiBody(baseMass=1, baseCollisionShapeIndex=collision, baseVisualShapeIndex=visual, basePosition=pos)
    objects.append(obj)

# Add some spheres
for i in range(3):
    radius = random.uniform(0.05, 0.1)
    pos = [random.uniform(-1, 1), random.uniform(-1, 1), radius + 1]  # Elevated initial position
    color = random_color()
    visual = p.createVisualShape(shapeType=p.GEOM_SPHERE, radius=radius, rgbaColor=color)
    collision = p.createCollisionShape(shapeType=p.GEOM_SPHERE, radius=radius)
    obj = p.createMultiBody(baseMass=1, baseCollisionShapeIndex=collision, baseVisualShapeIndex=visual, basePosition=pos)
    objects.append(obj)

# Add a cylinder
cylinder_radius = 0.1
cylinder_height = 0.3
cylinder_pos = [0.5, 0.5, cylinder_height/2 + 1]  # Elevated initial position
cylinder_color = random_color()
cylinder_visual = p.createVisualShape(shapeType=p.GEOM_CYLINDER, radius=cylinder_radius, length=cylinder_height, rgbaColor=cylinder_color)
cylinder_collision = p.createCollisionShape(shapeType=p.GEOM_CYLINDER, radius=cylinder_radius, height=cylinder_height)
cylinder = p.createMultiBody(baseMass=1, baseCollisionShapeIndex=cylinder_collision, baseVisualShapeIndex=cylinder_visual, basePosition=cylinder_pos)
objects.append(cylinder)

# Add a complex object (a duck)
duck_pos = [-0.5, -0.5, 1.1]  # Elevated initial position
duck = p.loadURDF("duck_vhacd.urdf", basePosition=duck_pos, globalScaling=0.5)
objects.append(duck)

# Create a box to represent the D435 camera
cameraSize = [0.09, 0.025, 0.025]  # Approximate size of D435
camera_color = [0.8, 0.8, 0.8, 1]  # Light gray color
visualShapeId = p.createVisualShape(shapeType=p.GEOM_BOX, halfExtents=cameraSize, rgbaColor=camera_color)
collisionShapeId = p.createCollisionShape(shapeType=p.GEOM_BOX, halfExtents=cameraSize)

# Set up virtual RGB and depth cameras with higher resolution
width, height = 1280, 720  # High resolution
fov = 60  # Reduced FOV for a more natural perspective
aspect = width / height
near = 0.1
far = 10

# Compute camera position and orientation for isometric-like view
camera_distance = 4
camera_height = 2
camera_target = [0, 0, 0]
camera_yaw = 30  # Rotate 45 degrees around z-axis
camera_pitch = -20  # Tilt down 30 degrees

# Convert yaw and pitch to radians
yaw_rad = math.radians(camera_yaw)
pitch_rad = math.radians(camera_pitch)

# Calculate camera position
cameraPos = [
    camera_distance * math.cos(yaw_rad) * math.cos(pitch_rad),
    camera_distance * math.sin(yaw_rad) * math.cos(pitch_rad),
    camera_height + camera_distance * math.sin(pitch_rad)
]

# Calculate camera orientation
cameraOrn = p.getQuaternionFromEuler([pitch_rad, 0, yaw_rad])

# Create the camera object at the new position
cameraId = p.createMultiBody(baseMass=0.1,
                             baseCollisionShapeIndex=collisionShapeId,
                             baseVisualShapeIndex=visualShapeId,
                             basePosition=cameraPos,
                             baseOrientation=cameraOrn)

# Compute view and projection matrices
viewMatrix = p.computeViewMatrix(cameraPos, camera_target, [0, 0, 1])
projectionMatrix = p.computeProjectionMatrixFOV(fov, aspect, near, far)

# Set up video writer
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
video_writer = cv2.VideoWriter('simulation_video_isometric.mp4', fourcc, 30.0, (width, height))

# Simulation loop
num_frames = 300  # 10 seconds at 30 fps
for frame in range(num_frames):
    p.stepSimulation()
    
    # Get RGB and depth images
    _, _, rgbImg, depthImg, _ = p.getCameraImage(
        width, height, viewMatrix, projectionMatrix, 
        renderer=p.ER_BULLET_HARDWARE_OPENGL
    )
    
    # Convert the rgbImg to a numpy array
    rgb_array = np.array(rgbImg, dtype=np.uint8).reshape(height, width, 4)
    rgb_array = rgb_array[:, :, :3]  # Remove alpha channel
    
    # Convert RGB to BGR (OpenCV uses BGR)
    bgr_array = cv2.cvtColor(rgb_array, cv2.COLOR_RGB2BGR)
    
    # Write the frame to video
    video_writer.write(bgr_array)
    
    # Print progress
    if frame % 30 == 0:
        print(f"Processed frame {frame}/{num_frames}")

# Release video writer
video_writer.release()

p.disconnect()

print("Simulation completed. Video saved as 'simulation_video_isometric.mp4'.")





import pybullet as p
import pybullet_data
import numpy as np
import cv2

# Connect to physics server
p.connect(p.DIRECT)

# Load environment (plane and robot)
p.setAdditionalSearchPath(pybullet_data.getDataPath())
plane = p.loadURDF("plane.urdf")
robot = p.loadURDF("r2d2.urdf", [0, 0, 0.5])  # Robot starts at (0, 0, 0.5)

# Setup camera parameters (RealSense D430 Depth Camera)
depth_width, depth_height = 1280, 720
rgb_width, rgb_height = 1920, 1080
depth_fov_h, depth_fov_v = 87, 58  # Horizontal and Vertical FOV for depth
rgb_fov_h, rgb_fov_v = 69, 42  # Horizontal and Vertical FOV for RGB

near = 0.28  # Min depth distance (28 cm)
far = 10.0   # Maximum depth range

# Camera position and orientation (can be changed)
camera_target = [0, 0, 0.5]  # Pointing towards the robot
camera_position = [2, 0, 1]  # Positioned behind and above the robot
camera_up_vector = [0, 0, 1]

# Set view matrix (this could be dynamically updated later)
view_matrix = p.computeViewMatrix(camera_position, camera_target, camera_up_vector)

# Projection matrix for Depth camera (based on FOV)
depth_projection_matrix = p.computeProjectionMatrixFOV(depth_fov_h, depth_width / depth_height, near, far)

# Projection matrix for RGB camera (based on FOV)
rgb_projection_matrix = p.computeProjectionMatrixFOV(rgb_fov_h, rgb_width / rgb_height, near, far)

# Simulation parameters
p.setGravity(0, 0, -9.8)
p.setRealTimeSimulation(0)  # Disable real-time simulation to manually control steps

# Set up video writers for RGB and Depth feeds
fourcc = cv2.VideoWriter_fourcc(*'XVID')
rgb_out = cv2.VideoWriter('rgb_feed.avi', fourcc, 30.0, (rgb_width, rgb_height))
depth_out = cv2.VideoWriter('depth_feed.avi', fourcc, 30.0, (depth_width, depth_height))

# Simulation time and step duration
simulation_time = 10  # Run the simulation for 10 seconds
steps_per_second = 1000 # Target steps per second
total_steps = simulation_time * steps_per_second

# Start simulation loop
for i in range(total_steps):
    # Apply velocity to move the robot forward
    p.resetBaseVelocity(robot, linearVelocity=[0, 10, 0], angularVelocity=[0, 0, 0])

    # Get robot position and orientation for camera
    robot_position, robot_orientation = p.getBasePositionAndOrientation(robot)
    
    # Update the camera to follow the robot (optional: dynamic camera)
    camera_target = robot_position
    camera_position = [camera_target[0] + 2, camera_target[1], camera_target[2] + 1]
    view_matrix = p.computeViewMatrix(camera_position, camera_target, camera_up_vector)

    # Get Depth camera image
    _, _, depth_rgb_img, depth_img, _ = p.getqqCameraImage(
        width=depth_width, height=depth_height, viewMatrix=view_matrix, projectionMatrix=depth_projection_matrix
    )

    # Get RGB camera image
    _, _, rgb_img, _, _ = p.getCameraImage(
        width=rgb_width, height=rgb_height, viewMatrix=view_matrix, projectionMatrix=rgb_projection_matrix
    )

    # Convert depth buffer to actual depth values (far and near plane adjustments)
    depth_img = np.reshape(depth_img, [depth_height, depth_width])
    depth = far * near / (far - (far - near) * depth_img)

    # Convert images to display and saving format
    depth_rgb_img = np.reshape(depth_rgb_img, (depth_height, depth_width, 4))[:, :, :3]  # Remove alpha channel
    rgb_img = np.reshape(rgb_img, (rgb_height, rgb_width, 4))[:, :, :3]

    # Write frames to the video files
    rgb_out.write(np.uint8(rgb_img))
    depth_normalized = (depth / np.max(depth) * 255).astype(np.uint8)  # Normalize depth for visualization
    depth_out.write(cv2.cvtColor(depth_normalized, cv2.COLOR_GRAY2BGR))  # Convert depth to BGR for video

    # Step simulation
    p.stepSimulation()

    # Allow breaking the loop early if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release video writers
rgb_out.release()
depth_out.release()

# Cleanup
cv2.destroyAllWindows()
p.disconnect()
