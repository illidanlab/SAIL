import gym

from stable_baselines import DQN
from stable_baselines import PPO2
from stable_baselines.gail import ExpertDataset

# Create environment
env = gym.make('CartPole-v1')
save_dir = "/mnt/research/linkaixi/AllData/rewardshaping/"
# Instantiate the agent
# model = DQN('MlpPolicy', env, learning_rate=1e-3, prioritized_replay=True, verbose=1)
# Train the agent
# model.learn(total_timesteps=int(2e5))
# Save the agent
# model.save(save_dir + "dqn_cartPole")
# del model  # delete trained model to demonstrate loading

# print(env.action_space.n)
# Load the trained agent
model = DQN.load("dqn_cartPole")

# Enjoy trained agent
obs = env.reset()
for i in range(1000):
    action, _states = model.predict(obs)
    obs, rewards, dones, info = env.step(action)
    # env.render()
    print(rewards)
