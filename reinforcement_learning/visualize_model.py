import time
import argparse
import pybullet as p
import os
from stable_baselines3 import SAC, PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from robotenv import HERRobotEnv

def print_camera_controls():
    """Print camera control instructions"""
    print("\nCamera Controls:")
    print("- Mouse Left Button + Drag: Rotate camera")
    print("- Mouse Right Button + Drag: Pan camera")
    print("- Mouse Wheel / Middle Button + Drag: Zoom camera")
    print("- Ctrl + Left Click: Select focus point")
    print("- R key: Reset camera to default view")
    print("\n")

def make_gui_env():
    """Create environment with GUI enabled"""
    def _init():
        # Override PyBullet mode to GUI
        orig_connect = p.connect
        p.connect = lambda mode: orig_connect(p.GUI)
        
        env = HERRobotEnv()
        
        # Restore original connect function
        p.connect = orig_connect
        
        # Configure debug visualization for better ball tracking
        p.configureDebugVisualizer(p.COV_ENABLE_GUI, 0)  # Disable GUI overlay
        p.configureDebugVisualizer(p.COV_ENABLE_SHADOWS, 1)  # Keep shadows for depth perception
        p.configureDebugVisualizer(p.COV_ENABLE_KEYBOARD_SHORTCUTS, 1)  # Enable keyboard shortcuts
        p.configureDebugVisualizer(p.COV_ENABLE_MOUSE_PICKING, 1)  # Enable mouse picking
        
        # Set initial camera view
        reset_camera()
        return env
    return _init

def reset_camera():
    """Reset camera to default position"""
    p.resetDebugVisualizerCamera(
        cameraDistance=3.0,
        cameraYaw=60,
        cameraPitch=-20,
        cameraTargetPosition=[0.75, 0.75, 1.0]
    )

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, required=True,
                      help="Path to the model file (SAC or PPO)")
    parser.add_argument("--algo", type=str, choices=['sac', 'ppo'], required=True,
                      help="Algorithm type (SAC or PPO)")
    parser.add_argument("--difficulty", type=int, default=3, choices=[0,1,2,3],
                      help="Curriculum difficulty (0-3)")
    parser.add_argument("--speed", type=float, default=0.25,
                      help="Simulation speed (1.0 = real-time, lower = slower)")
    args = parser.parse_args()

    # Create vectorized environment with GUI
    env = DummyVecEnv([make_gui_env()])

    # Try to find the stats file
    model_dir = os.path.dirname(args.model_path)
    possible_stats_files = [
        os.path.join(model_dir, "her_sac_robot_final.stats"),  # Same directory as model
        os.path.join(model_dir, "..", "her_sac_robot_final.stats"),  # Parent directory
        os.path.join(os.path.dirname(model_dir), "her_sac_robot_final.stats"),  # Parent of model directory
    ]
    
    stats_file = None
    for stats_path in possible_stats_files:
        if os.path.exists(stats_path):
            stats_file = stats_path
            break
    
    if stats_file is None:
        print("Warning: Could not find normalization stats file. Running without normalization.")
        env = VecNormalize(
            env,
            norm_obs=True,
            norm_reward=False,
            clip_obs=10.0,
            training=False
        )
    else:
        print(f"Loading normalization stats from: {stats_file}")
        env = VecNormalize.load(stats_file, env)
        env.training = False  # Don't update normalization statistics
        env.norm_reward = False

    # Set curriculum phase
    env.envs[0].curriculum_phase = args.difficulty
    print(f"\nRunning at difficulty level {args.difficulty}")
    print(f"Ball parameters for this phase:")
    print(f"Distance range: {env.envs[0].phase_descriptions[args.difficulty]}")

    # Load the model
    print(f"\nLoading {args.algo.upper()} model from {args.model_path}")
    if args.algo.lower() == 'sac':
        model = SAC.load(args.model_path, env=env)
    else:
        model = PPO.load(args.model_path, env=env)

    print("\nStarting visualization... Press Ctrl+C to stop")
    print("(Ball throws will be shown at {:.0f}% speed)".format(args.speed * 100))
    print_camera_controls()
    print("\nWaiting for first throw...")
    
    # Main visualization loop
    obs = env.reset()
    episode_num = 1
    try:
        while True:
            # Get action from model
            action, _ = model.predict(obs, deterministic=True)
            
            # Step environment with delay for visualization
            obs, reward, done, info = env.step(action)
            time.sleep((1/240) / args.speed)  # Slow down the simulation
            
            # Check for camera reset key (R)
            keys = p.getKeyboardEvents()
            if ord('r') in keys and keys[ord('r')] & p.KEY_WAS_TRIGGERED:
                reset_camera()
                print("Camera reset to default view")
            
            if done[0]:
                success = info[0].get('is_success', False)
                print(f"\nThrow {episode_num}: {'✓ CAUGHT!' if success else '✗ MISSED'}")
                episode_num += 1
                print("\nWaiting for next throw...")
                obs = env.reset()

    except KeyboardInterrupt:
        print("\nVisualization stopped by user")

    finally:
        env.close()

if __name__ == "__main__":
    main() 