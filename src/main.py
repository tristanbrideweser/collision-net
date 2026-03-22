import gymnasium as gym
import panda_gym
import numpy as np

# Use PandaReach-v3 as the base
env = gym.make("PandaReach-v3", render_mode="human")
observation, info = env.reset()

print("Environment initialized. Close the window to stop.")

try:
    while True:
        # Provide a neutral action (no movement)
        # For PandaReach, actions are usually 3D displacement
        action = np.zeros(env.action_space.shape) 
        
        # Step the simulation forward
        observation, reward, terminated, truncated, info = env.step(action)
        
        if terminated or truncated:
            observation, info = env.reset()
            
except KeyboardInterrupt:
    print("Shutting down...")
finally:
    env.close()