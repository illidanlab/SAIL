import argparse
import time
import difflib
import os
from collections import OrderedDict
from pprint import pprint
import warnings
import importlib
from settings import CONFIGS
import tensorflow as tf
warnings.filterwarnings("ignore")
import gym
try:
    import pybullet_envs
except ImportError:
    pybullet_envs = None
import numpy as np
import yaml
try:
    import highway_env
except ImportError:
    highway_env = None
from mpi4py import MPI

from stable_baselines.common import set_global_seeds
from stable_baselines.common.cmd_util import  make_atari_env_with_log_monitor
from stable_baselines.common.vec_env import VecFrameStack, VecNormalize, DummyVecEnv, VecEnv
from stable_baselines.ddpg import AdaptiveParamNoiseSpec, NormalActionNoise, OrnsteinUhlenbeckActionNoise
from stable_baselines.bench import Monitor
from stable_baselines.ppo2.ppo2 import constfn
from stable_baselines.results_plotter import load_results, ts2xy
from stable_baselines.gail import generate_expert_traj_mujoco
from stable_baselines.td3.lfd_envs import AbsorbingWrapper 

from utils import make_env_with_log_monitor, ALGOS, linear_schedule, get_wrapper_class
from utils.noise import LinearNormalActionNoise

best_mean_reward, n_steps = -np.inf, 0

def is_mujoco(env):
    for name in ['Hopper', 'Half', 'Walker', 'Reacher', 'Ant', 'Humanoid', 'InvertedDoublePendulum', 'Pusher', 'Swimmer', 'Thrower', 'Striker']:
        if name in env:
            return True
    return False

def check_if_atari(env_id):
    is_atari = False
    no_skip = False
    for game in ['Gravitar', 'MontezumaRevenge', 'Pitfall', 'Qbert', 'Pong']:
        if game in env_id:
            is_atari = True
            if 'NoFrameskip' in env_id:
                no_skip = True
            return is_atari, no_skip

def a2c_callback(log_dir, max_score, mode):
    def callback(_locals, _globals):
        """
        Callback called at each step (for DQN an others) or after n steps (see ACER or PPO2)
        :param _locals: (dict)
        :param _globals: (dict)
        """
        global n_steps, best_mean_reward
        # Print stats every 20 calls
        if (n_steps + 1) % 500 == 0:
            # Evaluate policy training performance
            x, y = ts2xy(load_results(log_dir), 'timesteps')
            if len(x) > 0:
                mean_reward = np.mean(y[-100:])
                print(x[-1], 'timesteps')
                print("Best mean reward: {:.2f} - Last mean reward per episode: {:.2f}".format(best_mean_reward, mean_reward))

                # New best model, you could save the agent here
                if mean_reward > best_mean_reward:
                    best_mean_reward = mean_reward
                    # Example for saving best model
                    print("Saving new best model")
                    _locals['self'].save(os.path.join(log_dir, 'best_model.pkl'))
                if 'train' in mode and mean_reward > max_score:
                    print("Stop training.")
                    return False
        n_steps += 1
        return True
    return callback

def load_expert_hyperparams(args):
    with open('../hyperparams/{}.yml'.format(args.algo), 'r') as f:
        hyperparams_dict = yaml.load(f)
        if env_id in list(hyperparams_dict.keys()):
            hyperparams = hyperparams_dict[env_id]
        elif is_atari:
            hyperparams = hyperparams_dict['atari']
        else:
            raise ValueError("Hyperparameters not found for {}-{}".format(args.algo, env_id))

    # Sort hyperparams that will be saved
    saved_hyperparams = OrderedDict([(key, hyperparams[key]) for key in sorted(hyperparams.keys())])

    algo_ = args.algo

    if args.verbose > 0:
        pprint(saved_hyperparams)
    return hyperparams, saved_hyperparams


def initiate_hyperparams(args, hyperparams):
    n_envs = hyperparams.get('n_envs', 1)

    if args.verbose > 0:
        print("Using {} environments".format(n_envs))

    # Create learning rate schedules for ppo2 and sac
    if args.algo in ["ppo2", "sac", "td3", "dac", "sail", "dualsail", "bcq", "sdice", "dice"]:
        for key in ['learning_rate', 'cliprange', 'cliprange_vf']:
            if key not in hyperparams:
                continue
            if isinstance(hyperparams[key], str):
                schedule, initial_value = hyperparams[key].split('_')
                initial_value = float(initial_value)
                hyperparams[key] = linear_schedule(initial_value)
            elif isinstance(hyperparams[key], (float, int)):
                # Negative value: ignore (ex: for clipping)
                if hyperparams[key] < 0:
                    continue
                hyperparams[key] = constfn(float(hyperparams[key]))
            else:
                raise ValueError('Invalid value for {}: {}'.format(key, hyperparams[key]))
    return hyperparams

def create_env_2(args, hyperparams, n_envs, env_wrapper=None, normalize=False, sparse_wrapper=None, normalize_kwargs={}):
    #
    if is_atari:
        if args.verbose > 0:
            print("Using Atari wrapper")
        #env = make_atari_env(env_id, num_env=n_envs, seed=args.seed)
        env = make_atari_env_with_log_monitor(env_id, log_dir=args.log_dir, num_env=n_envs, seed=args.seed)
        # Frame-stacking with 4 frames
        env = VecFrameStack(env, n_stack=4)
    elif args.algo in ['dqn', 'ddpg', 'dqnrrs']:
        env = gym.make(env_id)
        env.seed(args.seed)
        env = Monitor(env, args.log_dir, allow_early_resets=True)
        ## add sparse environment wrapper after the monitor wrapper
        #env = CartPoleWrapper(env)
        if env_wrapper is not None:
            print("Using Predefined Env Wrapper")
            env = env_wrapper(env)
        env = DummyVecEnv([lambda: env])
    else: # ppo2, trpo
        if env_wrapper is not None:
            print("Using Predefined Env Wrapper")
        if n_envs == 1:
            env = make_env_with_log_monitor(
                env_id,
                0,
                args.seed,
                log_dir=args.log_dir,
                wrapper_class=[env_wrapper,AbsorbingWrapper] if args.algo in ['dac'] else [env_wrapper])
            env = DummyVecEnv([env])
        else:
            # On most env, SubprocVecEnv does not help and is quite memory hungry
            env = DummyVecEnv([
                make_env_with_log_monitor(
                    env_id,
                    i,
                    args.seed,
                    log_dir=args.log_dir,
                    wrapper_class=[env_wrapper,AbsorbingWrapper] if args.algo in ['dac'] else [env_wrapper],
                    sparse_wrapper=sparse_wrapper) for i in range(n_envs)])
        if normalize:
            if args.verbose > 0:
                if len(normalize_kwargs) > 0:
                    print("Normalization activated: {}".format(normalize_kwargs))
                else:
                    print("Normalizing input and reward")
            env = VecNormalize(env, **normalize_kwargs)
    # Optional Frame-stacking
    if hyperparams.get('frame_stack', False):
        n_stack = hyperparams['frame_stack']
        env = VecFrameStack(env, n_stack)
        print("Stacking {} frames".format(n_stack))
        del hyperparams['frame_stack']
    return env, hyperparams


def create_test_env(args, hyperparams):
    env_wrapper = get_wrapper_class(hyperparams)
    n_envs = hyperparams.get('n_envs', 1)
    print("Creating Testing environment for {} ....".format(args.env))

    if env_wrapper is not None:
        print("Using Predefined Env Wrapper")
    logdir = os.path.join(args.log_dir, 'test')
    if n_envs == 1:
        env = make_env_with_log_monitor(
            env_id,
            0,
            args.seed,
            log_dir=logdir, 
            wrapper_class=[env_wrapper,AbsorbingWrapper] if args.algo in ['dac'] else [env_wrapper])
        env = env()
    return env

def create_env(args, hyperparams, n_timesteps):
    normalize = False
    normalize_kwargs = {}
    if 'normalize' in hyperparams.keys():
        normalize = hyperparams['normalize']
        if isinstance(normalize, str):
            normalize_kwargs = eval(normalize)
            normalize = True
        del hyperparams['normalize']

    if 'policy_kwargs' in hyperparams.keys():
        hyperparams['policy_kwargs'] = eval(hyperparams['policy_kwargs'])

    # Delete keys so the dict can be passed to the model constructor
    if 'n_envs' in hyperparams.keys():
        del hyperparams['n_envs']
    del hyperparams['n_timesteps']
    # obtain a class object from a wrapper name string in hyperparams
    # and delete the entry
    env_wrapper = get_wrapper_class(hyperparams)


    n_envs = hyperparams.get('n_envs', 1)
    print("Creating Sparse environment for {} ....".format(args.env))
    sparse_wrapper=None

    env, hyperparams = create_env_2(
        args,
        hyperparams,
        n_envs,
        env_wrapper=env_wrapper,
        normalize=normalize,
        normalize_kwargs=normalize_kwargs)
    # Stop env processes to free memory
    if args.optimize_hyperparameters and n_envs > 1:
        env.close()

    # Parse noise string for DDPG and SAC
    if args.algo in ['ddpg', 'sac', 'td3', "sail", "dualsail", "bcq", "sdice", "dice", "dac"] and hyperparams.get('noise_type') is not None:
        noise_type = hyperparams['noise_type'].strip()
        noise_std = hyperparams['noise_std']
        n_actions = env.action_space.shape[0]
        if 'adaptive-param' in noise_type:
            assert args.algo == 'ddpg', 'Parameter is not supported by SAC'
            hyperparams['param_noise'] = AdaptiveParamNoiseSpec(initial_stddev=noise_std,
                                                                desired_action_stddev=noise_std)
        elif 'normal' in noise_type:
            if 'lin' in noise_type:
                hyperparams['action_noise'] = LinearNormalActionNoise(mean=np.zeros(n_actions),
                                                                      sigma=noise_std * np.ones(n_actions),
                                                                      final_sigma=hyperparams.get('noise_std_final', 0.0) * np.ones(n_actions),
                                                                      max_steps=n_timesteps)
            else:
                hyperparams['action_noise'] = NormalActionNoise(mean=np.zeros(n_actions),
                                                                sigma=noise_std * np.ones(n_actions))
                # hyperparams["batch_action_noise"] = BatchNormalActionNoise(mean=np.zeros(n_actions),
                #                                                 sigma=noise_std*np.ones(n_actions),
                #                                                 hyperparams['kernel_dim'])
        elif 'ornstein-uhlenbeck' in noise_type:
            hyperparams['action_noise'] = OrnsteinUhlenbeckActionNoise(mean=np.zeros(n_actions),
                                                                       sigma=noise_std * np.ones(n_actions))
        else:
            raise RuntimeError('Unknown noise type "{}"'.format(noise_type))
        print("Applying {} noise with std {}".format(noise_type, noise_std))
        del hyperparams['noise_type']
        #del hyperparams['noise_std']
        if 'noise_std_final' in hyperparams:
            del hyperparams['noise_std_final']
    return env, hyperparams, normalize


def eval_mujoco_model(args, model, env, step=50000):
    ###########################
    # Force deterministic for DQN, DDPG, SAC and HER (that is a wrapper around)
    deterministic = args.algo in ['dqn', 'ddpg', 'sac', 'her', 'td3', "sail", "dualsail", "bcq", "dice", "sdice", "dac", 'dqnrrs']
    episode_reward = 0.0
    episode_rewards = []
    ep_len = 0
    # For HER, monitor success rate
    obs = env.reset()
    successes = []


    for i in range(step):
        action, _ = model.predict(obs, deterministic=deterministic)
        action_prob = model.action_probability(obs)
        #if i % 1000 == 0:
        #    print(action_prob)
        # Clip Action to avoid out of bound errors
        if isinstance(env.action_space, gym.spaces.Box):
            action = np.clip(action, env.action_space.low, env.action_space.high)
        obs, reward, done, infos = env.step(action)
        if not args.no_render:
            env.render('human')

        ep_len += 1

        n_envs = hyperparams.get('n_envs', 1)
        if n_envs == 1:
            if done and not is_atari and args.verbose > 0:
                x, y = ts2xy(load_results(args.log_dir), 'timesteps')
                if len(x) > 0:
                    mean_reward = np.mean(y[-100:])
                    episode_reward = y[-1]
                    print(x[-1], 'timesteps')
                    print("Length: {}, Mean reward: {:.2f} - Last episode reward: {:.2f}".format(ep_len, mean_reward, episode_reward))
                episode_rewards.append(episode_reward)
                ep_len = 0


    if args.verbose > 0 and len(episode_rewards) > 0:
        print("Mean reward: {:.2f}".format(np.mean(episode_rewards)))

    # Workaround for https://github.com/openai/gym/issues/893
    if not args.no_render:
        if args.n_envs == 1 and 'Bullet' not in env_id and not is_atari and isinstance(env, VecEnv):
            # DummyVecEnv
            # Unwrap env
            while isinstance(env, VecNormalize) or isinstance(env, VecFrameStack):
                env = env.venv
            env.envs[0].env.close()
        else:
            env.close()

def eval_model(args, model, env, step=50000):
    #exp_folder = args.trained_agent.split('.pkl')[0]
    #if normalize:
    #    print("Loading saved running average")
    #    env.load_running_average(exp_folder)

    ###########################
    # Force deterministic for DQN, DDPG, SAC and HER (that is a wrapper around)
    deterministic = args.algo in ['dqn', 'ddpg', 'sac', 'her', 'dac', 'td3', 'sail', 'dualsail', 'bcq', "dice", 'sdice', 'dqnrrs']
    episode_reward = 0.0
    episode_rewards = []
    ep_len = 0
    # For HER, monitor success rate
    obs = env.reset()
    successes = []
    for i in range(step):
        action, _ = model.predict(obs, deterministic=deterministic)
        action_prob = model.action_probability(obs)
        if i % 1000 == 0:
            print(action_prob)
        # Random Agent
        # action = [env.action_space.sample()]
        # Clip Action to avoid out of bound errors
        if isinstance(env.action_space, gym.spaces.Box):
            action = np.clip(action, env.action_space.low, env.action_space.high)
        obs, reward, done, infos = env.step(action)
        if not args.no_render:
            env.render('human')

        episode_reward += reward[0]
        ep_len += 1

        n_envs = hyperparams.get('n_envs', 1)
        if n_envs == 1:
            # For atari the return reward is not the atari score
            # so we have to get it from the infos dict
            if is_atari and infos is not None and args.verbose >= 1:
                episode_infos = infos[0].get('episode')
                if episode_infos is not None:
                    print("Atari Episode Score: {:.2f}".format(episode_infos['r']))
                    print("Atari Episode Length", episode_infos['l'])

            if done and not is_atari and args.verbose > 0:
                # NOTE: for env using VecNormalize, the mean reward
                # is a normalized reward when `--norm_reward` flag is passed
                print("Episode Reward: {:.2f}".format(episode_reward))
                print("Episode Length", ep_len)
                episode_infos = infos[0].get('episode')
                if episode_infos is not None:
                    print("Episode Reward: {}".format(episode_infos))
                episode_rewards.append(episode_reward)
                episode_reward = 0.0
                ep_len = 0

            # Reset also when the goal is achieved when using HER
            if done or infos[0].get('is_success', False):
                if args.algo == 'her' and args.verbose > 1:
                    print("Success?", infos[0].get('is_success', False))
                # Alternatively, you can add a check to wait for the end of the episode
                # if done:
                obs = env.reset()
                if args.algo == 'her':
                    successes.append(infos[0].get('is_success', False))
                    episode_reward, ep_len = 0.0, 0

    if args.verbose > 0 and len(successes) > 0:
        print("Success rate: {:.2f}%".format(100 * np.mean(successes)))

    if args.verbose > 0 and len(episode_rewards) > 0:
        print("Mean reward: {:.2f}".format(np.mean(episode_rewards)))

    # Workaround for https://github.com/openai/gym/issues/893
    if not args.no_render:
        if args.n_envs == 1 and 'Bullet' not in env_id and not is_atari and isinstance(env, VecEnv):
            # DummyVecEnv
            # Unwrap env
            while isinstance(env, VecNormalize) or isinstance(env, VecFrameStack):
                env = env.venv
            env.envs[0].env.close()
        else:
            env.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--env', type=str, default="PongNoFrameskip-v4", help='environment ID')
    parser.add_argument('-tb', '--tensorboard-log', help='Tensorboard log dir', default='tb_logs', type=str)
    parser.add_argument('-i', '--trained-agent', help='Path to a pretrained agent to continue training',
                        default='', type=str)
    parser.add_argument('--algo', help='RL Algorithm', default='dqnrrs',
                        type=str, required=False, choices=list(ALGOS.keys()))
    parser.add_argument('-n', '--n-timesteps', help='Overwrite the number of timesteps', default=-1,
                        type=int)
    parser.add_argument('--log-interval', help='Override log interval (default: -1, no change)', default=1000,
                        type=int)
    parser.add_argument('-f', '--log-folder', help='Log folder', type=str, default='logs')
    parser.add_argument('--seed', help='Random generator seed', type=int, default=1)
    parser.add_argument('--n-trials', help='Number of trials for optimizing hyperparameters', type=int, default=10)
    parser.add_argument('--log-dir', help='Log directory', type=str, default='/tmp/logs') # required=True,
    parser.add_argument('-optimize', '--optimize-hyperparameters', action='store_true', default=False,
                        help='Run hyperparameters search')
    parser.add_argument('--n-jobs', help='Number of parallel jobs when optimizing hyperparameters', type=int, default=6)
    parser.add_argument('--n-episodes', help='Number of expert episodes', type=int, default=1)
    parser.add_argument('--no-render', help='If render', default=True)
    parser.add_argument('--sampler', help='Sampler to use when optimizing hyperparameters', type=str,
                        default='skopt', choices=['random', 'tpe', 'skopt'])
    parser.add_argument('--pruner', help='Pruner to use when optimizing hyperparameters', type=str,
                        default='none', choices=['halving', 'median', 'none'])
    parser.add_argument('--verbose', help='Verbose mode (0: no output, 1: INFO)', default=1,
                        type=int)
    parser.add_argument('--gym-packages', type=str, nargs='+', default=[], help='Additional external Gym environemnt package modules to import (e.g. gym_minigrid)')
    parser.add_argument('--eval-model-path', type=str, default='', help="model path to be evaluated (when task == 'eval')")
    parser.add_argument(
        '--task',
        type=str,
        default='train',
        help='Figure it out yourself.╭(╯^╰)╮')
    args = parser.parse_args()

    # build log_dir
    # $prefix/$task/$algo/$env/rank$seed"
    log_dir = os.path.join(args.log_dir, args.task, args.algo, args.env, "rank{}".format(args.seed))
    os.system("mkdir -p {}".format(log_dir))
    args.orig_log_dir = args.log_dir
    args.log_dir = log_dir

    # Going through custom gym packages to let them register in the global registory
    for env_module in args.gym_packages:
        importlib.import_module(env_module)

    env_id = args.env
    registered_envs = set(gym.envs.registry.env_specs.keys())

    # If the environment is not found, suggest the closest match
    if args.env not in registered_envs:
        try:
            closest_match = difflib.get_close_matches(args.env, registered_envs, n=1)[0]
        except IndexError:
            closest_match = "'no close match found...'"
        raise ValueError('{} not found in gym registry, you maybe meant {}?'.format(env_id, closest_match))

    set_global_seeds(args.seed)

    if args.trained_agent != "":
        #args.trained_agent = os.path.join(args.trained_agent, args.algo, "{}.pkl".format(args.env))
        print(args.trained_agent)
        assert args.trained_agent.endswith('.pkl') and os.path.isfile(args.trained_agent), \
            "The trained_agent must be a valid path to a .pkl file"

    rank = 0
    if MPI.COMM_WORLD.Get_size() > 1:
        print("Using MPI for multiprocessing with {} workers".format(MPI.COMM_WORLD.Get_size()))
        rank = MPI.COMM_WORLD.Get_rank()
        print("Worker rank: {}".format(rank))

        args.seed += rank
        if rank != 0:
            args.verbose = 0
            args.tensorboard_log = ''

    tensorboard_log = os.path.join(args.log_dir, 'tb')

    is_atari = False
    if 'NoFrameskip' in env_id:
        is_atari = True

    print("=" * 10, env_id, "=" * 10)

    #############################################
    # Load hyperparameters & create environment
    #############################################
    hyperparams, saved_hyperparams = load_expert_hyperparams(args)
    # replace str in hyperparams to be real entities
    hyperparams = initiate_hyperparams(args, hyperparams)
    # Should we overwrite the number of timesteps?
    if args.n_timesteps > 0:
        if args.verbose:
            print("Overwriting n_timesteps with n={}".format(args.n_timesteps))
        n_timesteps = args.n_timesteps
    else:
        n_timesteps = int(hyperparams['n_timesteps'] * 1.0 )
    print(n_timesteps)

    config = CONFIGS[args.env]
    config['n_episodes'] = args.n_episodes
    config['sparse'] = False
    #args.log_dir = args.log_dir.replace('eval-bc', 'eval-bc-episode-{}'.format(config['n_episodes']))  
    os.makedirs(args.log_dir, exist_ok=True)
    env, hyperparams, normalize = create_env(args, hyperparams, n_timesteps)

    ######################################
    # build a new model or a trained agent
    ######################################

    policy = hyperparams['policy']
    del hyperparams['policy']


    hyperparams['using_gail'] = 'gail' in args.task or 'airl' in args.task
    if args.algo in ["sail", "dac", "sdice"]:
        hyperparams['using_return'] = 'return' in args.task
        hyperparams['bc_expert'] = 'expert' in args.task
        hyperparams['lfd'] = 'lfd' in args.task
        hyperparams['adaptive'] = 'adaptive' in args.task
        hyperparams['max_episode_n'] = args.n_episodes
        hyperparams['n_jobs'] = args.n_jobs
        hyperparams['max_episode_n'] = args.n_episodes
        if 'explore' in args.task:
            if 'uniform' in args.task:
                hyperparams['explore_mode'] = "uniform"
            else:
                hyperparams['explore_mode'] = "uniform"
        else:
            hyperparams['explore_mode'] = None
    if args.algo in ["sail"]:
        hyperparams['sparse_reward'] = 'sparse' in args.task
        hyperparams['norm_sparse_reward'] = 'sparse' in args.task #and 'norm' in args.task

    if args.algo in ["sac"]:
        hyperparams['adaptive'] = 'adaptive' in args.task
        hyperparams['lfd'] = 'lfd' in args.task
        hyperparams['n_jobs'] = args.n_jobs
        hyperparams['shaping_mode'] = args.task


    config['gail'] = hyperparams['using_gail']
    config['shaping_mode'] = args.task

    ##################
    # get path to expert demonstration data
    ##################
    data_save_dir = os.path.join("../../teacher_dataset", "expert_data_no_img_{}_scores_{}_episodes_{}".format(args.env.split('-')[0], config['optimal_score'], config['n_episodes']))
    data_save_dir = '{}.npz'.format(data_save_dir)
    print("Demo Data save in : {}".format(data_save_dir))

    #####################################
    # build model
    #####################################

    # Train an agent from scratch
    policy = 'MlpPolicy'

    policy_kwargs = dict(act_fun=tf.nn.tanh, layers=[64, 64, 64])
    kwargs = {}
    if args.log_interval > -1:
        kwargs = {'log_interval': args.log_interval}
    if 'test' in args.task:
        test_env = create_test_env(args, hyperparams)
    else:
        test_env = None
    if 'env_wrapper' in hyperparams.keys():
        del hyperparams['env_wrapper']

    model = ALGOS[args.algo](
        policy,
        env=env,
        tensorboard_log=tensorboard_log,
        verbose=args.verbose,
        expert_data_path=data_save_dir,
        **hyperparams)
    print("Model buit successfully.")
    ##############################################################
    # begin learning, saving best model through each call back
    ##############################################################
    if 'eval' in args.task:
        model_path = args.task.replace('eval-', '')
        args.eval_model_path=os.path.join(args.orig_log_dir, model_path, args.algo, args.env,
                                          "rank{}".format(args.seed),
                                          "best_model.pkl")  # this model is trained from scratch
        print('*'*40 + args.eval_model_path)

        model_file = args.eval_model_path #os.path.join(args.eval_model_path, 'best_model.pkl')
        assert os.path.isfile(model_file), \
            " The eval_model_path must be a valid path to a pkl or zip file, but no file found at {} ".format(model_file)
        print(model_file)
        model = ALGOS[args.algo].load(
            model_file, env=env,
            verbose=args.verbose)
        if is_mujoco(args.env):
            eval_mujoco_model(args, model, env)
            #eval_model(args, model, env)
        else:
            eval_model(args, model, env)

    elif 'pretrain' == args.task:
        if not os.path.isfile(data_save_dir):
            # if expert data is not ready, generate data first
            model_file = os.path.join("{}/train/{}".format(args.orig_log_dir, args.algo),args.env, "rank{}".format(args.seed), "best_model.pkl")
            pretrain_model = ALGOS[args.algo].load(
                model_file, env=env,
                verbose=args.verbose)

            is_atari = 'NoFrameskip' in args.env
            #generate_expert_traj_pro(
            generate_expert_traj_mujoco(
                pretrain_model,
                env=env,
                save_path=data_save_dir.replace('.npz',''),
                n_episodes=config['n_episodes'] ,
                optimal_score=config['optimal_score']
                )
        print("Demo Data save in : {}".format(data_save_dir))
        env.close()
    elif 'bc' == args.task:
        model.pretrain_with_buffer(
            config,
            args.log_dir,
            n_epochs=100 * config['n_episodes'],
            learning_rate=1e-3,
            adam_epsilon=1e-8,
            val_interval=None,
            )
    else:
        print(config)
        time.sleep(3)
        config['log_dir'] = args.log_dir
        cb_func = a2c_callback(args.log_dir, config['max_score'], args.task)
        model.learn(
            n_timesteps,
            callback=cb_func,
            config=config,
            test_env=test_env,
            **kwargs)
        with open(os.path.join(args.log_dir, 'config.yml'), 'w') as f:
            yaml.dump(saved_hyperparams, f)












