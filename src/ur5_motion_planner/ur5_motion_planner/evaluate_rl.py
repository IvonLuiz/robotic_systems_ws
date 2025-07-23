import rclpy
import numpy as np
import os
import time
import argparse
import pandas as pd

from stable_baselines3 import SAC, TD3, DDPG, PPO, A2C

from .env_node import UR5Env
from .config_loader import RLConfig

def evaluate_model(model_path, episodes=10):
    rclpy.init()

    config = RLConfig()
    env = UR5Env()

    algorithm_name = config.get('model.algorithm', 'SAC')
    algorithms = {
        'SAC': SAC,
        'TD3': TD3,
        'DDPG': DDPG,
        'PPO': PPO,
        'A2C': A2C
    }

    if algorithm_name not in algorithms:
        raise ValueError(f"Unsupported algorithm: {algorithm_name}")

    AlgorithmClass = algorithms[algorithm_name]

    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")

    print(f"🔍 Loading model: {model_path}")
    model = AlgorithmClass.load(model_path, env=env)

    results = []

    for ep in range(episodes):
        obs, _ = env.reset()
        target_position = env.target_position.copy()
        done = False
        steps = 0

        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            steps += 1

        final_position = env.get_end_effector_pose()[:3]
        final_distance = np.linalg.norm(final_position - target_position)

        print(f"\n🎯 Episode {ep+1}/{episodes}")
        print(f"Target Position  : {target_position}")
        print(f"Final EE Position: {final_position}")
        print(f"Final Distance   : {final_distance:.4f} m in {steps} steps")

        results.append({
            "episode": ep+1,
            "target": target_position,
            "final_position": final_position,
            "final_distance": final_distance,
            "steps": steps
        })

        time.sleep(1.0)

    print("\n=== Evaluation Summary ===")
    for r in results:
        print(f"Ep {r['episode']}: Distance = {r['final_distance']:.4f} m | Steps = {r['steps']}")

    csv_filename = "rl_evaluation_results.csv"
    df = pd.DataFrame([{
        "episode": r["episode"],
        "target_x": r["target"][0],
        "target_y": r["target"][1],
        "target_z": r["target"][2],
        "final_x": r["final_position"][0],
        "final_y": r["final_position"][1],
        "final_z": r["final_position"][2],
        "final_distance": r["final_distance"],
        "steps": r["steps"]
    } for r in results])

    df.to_csv(csv_filename, index=False)
    print(f"\n📁 Resultados salvos em: {csv_filename}")

    rclpy.shutdown()

    return results

def main():
    parser = argparse.ArgumentParser(description="Evaluate trained RL model for UR5.")
    parser.add_argument("--model_path", required=True, help="Path to the trained model .zip file")
    parser.add_argument("--episodes", type=int, default=10, help="Number of evaluation episodes")
    args = parser.parse_args()

    evaluate_model(args.model_path, episodes=args.episodes)

if __name__ == "__main__":
    main()
