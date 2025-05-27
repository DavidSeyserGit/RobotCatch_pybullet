import time
import pybullet as p
from stable_baselines3 import SAC
from stable_baselines3.common import utils
from robotenv import HERRobotEnv

def make_gui_env():
    orig_connect = p.connect
    p.connect = lambda mode: orig_connect(p.GUI)
    env = HERRobotEnv()
    p.connect = orig_connect
    return env

if __name__ == "__main__":
    # Monkey patch the space checking function to do nothing
    original_check = utils.check_for_correct_spaces
    utils.check_for_correct_spaces = lambda env, obs_space, action_space: None
    
    env = make_gui_env()
    
    p.resetDebugVisualizerCamera(
        cameraDistance=2,
        cameraYaw=45,
        cameraPitch=-30,
        cameraTargetPosition=[1, 0.75, 1],
    )

    model = SAC.load("her_sac_robot.zip", env=env)
    
    # Restore the original function (optional)
    utils.check_for_correct_spaces = original_check

    obs, _ = env.reset()
    while True:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, done, _, info = env.step(action)

        time.sleep(1.0 / 240.0)

        if done:
            print("Episode done. Success =", info.get("is_success", False))
            obs, _ = env.reset()