import pybullet as p
import pybullet_data
import time
import random

class Ball:
    def __init__(self, start_position, velocity_vector, radius=0.01, mass=0.07):
        self.start_position = start_position
        self.radius = radius
        self.mass = mass
        self.id = None
        self.velocity_vector = velocity_vector  # Store velocity vector here
        self.ground_contact_marked = False
        self.ground_contact_point = None

    def spawn(self):
        visualShapeId = p.createVisualShape(shapeType=p.GEOM_SPHERE, radius=self.radius, rgbaColor=[1, 0, 0, 1])
        collisionShapeId = p.createCollisionShape(shapeType=p.GEOM_SPHERE, radius=self.radius)
        self.id = p.createMultiBody(baseMass=self.mass, baseVisualShapeIndex=visualShapeId,
                                    baseCollisionShapeIndex=collisionShapeId, basePosition=self.start_position)
        p.resetBaseVelocity(self.id, linearVelocity=self.velocity_vector)

    def draw_velocity_vector(self):
        scale = 1
        end_pos = [self.start_position[i] + self.velocity_vector[i] * scale for i in range(3)]
        p.addUserDebugLine(self.start_position, end_pos, [1, 0.7, 0], 2, lifeTime=0)

    def check_ground_contact(self):
        if not self.ground_contact_marked:
            pos, _ = p.getBasePositionAndOrientation(self.id)
            if pos[2] <= self.radius:
                self.ground_contact_point = (pos[0], pos[1], 0)
                p.addUserDebugPoints([self.ground_contact_point], [[0, 0, 1]], pointSize=5, lifeTime=0)
                self.ground_contact_marked = True
                print(f"Ball touched the ground at position: x={self.ground_contact_point[0]:.2f}, "
                      f"y={self.ground_contact_point[1]:.2f}, z={self.ground_contact_point[2]:.2f}")
                
    def remove(self):
        if self.id is not None:
            p.removeBody(self.id)
            self.id = None


class Simulation:
    def __init__(self):
        p.connect(p.GUI)
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        p.setGravity(0, 0, -9.81)
        p.loadURDF("plane.urdf")
        self.balls = []

    def add_ball(self, ball):
        self.balls.append(ball)
        ball.spawn()
        ball.draw_velocity_vector()

    def run(self, simulation_time=5.0, time_step=1/250):
        steps = int(simulation_time / time_step)
        for step in range(steps):
            p.stepSimulation()
            for ball in self.balls:
                ball.check_ground_contact()
                if step % 10 == 0:
                    pos, _ = p.getBasePositionAndOrientation(ball.id)
            time.sleep(time_step)



if __name__ == "__main__":
    sim = Simulation()

    # Set common x velocity and y velocity
    x_velocity = 3  # Common x velocity for all balls
    y_velocity = 0  # y velocity is 0 for all balls

    # Create and add multiple balls
    num_balls = 15  # Number of balls to create
    for _ in range(num_balls):
        z_velocity = random.uniform(-2, 2)  # Random z velocity between -2 and 2
        ball = Ball(
            start_position=(0, 0, 1),  # All balls start at the same position
            velocity_vector=(x_velocity, y_velocity, z_velocity)
        )
        sim.add_ball(ball)

    sim.run(simulation_time=5.0)