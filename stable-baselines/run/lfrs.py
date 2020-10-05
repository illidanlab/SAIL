import os
import argparse
import numpy as np
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import gym

from stable_baselines import PPO2, A2C, ACER, ACKTR, DQN, HER, DDPG, TRPO, SAC, TD3
from stable_baselines.common.vec_env import DummyVecEnv, VecFrameStack, SubprocVecEnv 
from stable_baselines.common.cmd_util import make_atari_env, make_atari_env_no_skip
from stable_baselines.bench import Monitor
from settings import RS_LOGDIRS, NON_RS_LOGDIRS, PATH_PREFIX
from stable_baselines.results_plotter import load_results, ts2xy
from stable_baselines.logger import configure
from utils import create_test_env

POLICIES = {
    'A2C': A2C,
    'PPO2': PPO2,
    'DQN': DQN
}
best_mean_reward, n_steps = -np.inf, 0
def callback(_locals, _globals):
  """
  Callback called at each step (for DQN an others) or after n steps (see ACER or PPO2)
  :param _locals: (dict)
  :param _globals: (dict)
  """
  global n_steps, best_mean_reward
  # Print stats every 1000 calls
  if (n_steps + 1) % 1000 == 0:
      # Evaluate policy training performance
      x, y = ts2xy(load_results(log_dir), 'timesteps')
      if len(x) > 0:
          mean_reward = np.mean(y[-100:])
          print(x[-1], 'timesteps')
          print("Best mean reward: {:.2f} - Last mean reward per episode: {:.2f}".format(best_mean_reward, mean_reward))

          # # New best model, you could save the agent here
          if mean_reward > best_mean_reward:
              best_mean_reward = mean_reward
  n_steps += 1
  return True

def main(args):
    pass

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', help='Random generator seed', type=int, default=0)
    parser.add_argument(
        '--env', 
        help='Environment name', 
        type=str, 
        default='PongNoFrameskip-v4')
        #default='CartPole-v1')
    parser.add_argument(
        '--algo', 
        help='Algorithm used for reward shaping', 
        choices=['A2C', 'DQN'],
        type=str, 
        default='A2C')
        #default='DQN')
    parser.add_argument(
        '--log_dir', 
        help='model logging/saving directory',
        required=True)
    parser.add_argument(
        '--net',
        help='network structure',
        #default='CnnPolicy',
        default='MlpPolicy',
        choices=['MlpPolicy', 'CnnPolicy']
    )
    parser.add_argument(
        '--num_env',
        help="Number of envs",
        default=4,
        type=int
    )
    parser.add_argument(
        '--rank',
        help="rank id of experiments",
        default=0,
        type=int
    )
    parser.add_argument(
        '--reward_shaping',
        help="Whether to run reward shaping",
        default=1,
        choices=[0,1],
        type=int
    )
    parser.add_argument(
        '--temperature',
        help='temperature added for logit before softmax',
        default="1e2",
        type=str
        
    )
    args = parser.parse_args()
    args.temperature = float(args.temperature)


    if args.reward_shaping == 1:
        print("Running reward shaping ...")
        ##### set up environment and seed
        # log_dir = args.log_dir
        # if not log_dir:
        #     log_dir = RS_LOG_DIRS[args.algo]
        # log_dir = os.path.join(log_dir, "rank{}".format(args.rank))
        os.makedirs(args.log_dir, exist_ok=True)

        #configure()

        seed = args.seed
        ############################################################
        # create env, for now this only works for atari-games
        ############################################################

        if 'NoFrameskip' in args.env:
            env = make_atari_env(args.env, num_env=args.num_env, seed=seed)
            # Frame-stacking with 4 frames
            env = VecFrameStack(env, n_stack=4)
        else:
            env = make_atari_env_no_skip(args.env, num_env=args.num_env, seed=seed, log_dir=args.log_dir)
                

        # expert-model, 
        expert_model_path = "../trained_agents/{}/{}.pkl".format(args.algo.lower(), args.env)
        expert_model=POLICIES[args.algo].load(expert_model_path)
        expert_model.set_temperature(args.temperature)

        # reward-shaping model, use the same env as expert_model 
        rs_model = POLICIES[args.algo](args.net, env, verbose=1)
        #rs_model = POLICIES[args.algo](args.net, env, verbose=1, tensorboard_log=log_dir) 
    
        # begin learning from reward shaping
        if args.algo == 'A2C':
            rs_model.learn_from_rewardshaping(
                total_timesteps=int(1e7), 
                seed=seed, 
                expert_model=expert_model)
        else:
            rs_model.learn_from_rewardshaping(
                total_timesteps=int(1e7), 
                seed=seed, 
                expert_model=expert_model)
        save_path = os.path.join(args.log_dir, 'saved_model')
        rs_model.save(save_path)
    else:
        # non reward-shaping model
        print("Running training from scratch ...")
        ##### set up environment and seed
        log_dir = args.log_dir
        os.makedirs(log_dir, exist_ok=True)

        seed = args.seed
        ############################################################
        # create env 
        ############################################################
        ### LFRS
        env = gym.make(args.env)
        env = Monitor(env, log_dir, allow_early_resets=True)
        env = DummyVecEnv([lambda: env])
                
        model = POLICIES[args.algo]('MlpPolicy', env, verbose=1, tensorboard_log=log_dir)
        model.learn(
            total_timesteps=int(1e6), 
            seed=seed,
            callback=callback
        )
        save_path = os.path.join(args.log_dir, 'saved_model')
        model.save(save_path)
         
