import os
import numpy as np
import matplotlib.pyplot as plt

from stable_baselines.results_plotter import load_results, ts2xy
from settings import PATH_PREFIX

def moving_average(values, window):
    """
    Smooth values by doing a moving average
    :param values: (numpy array)
    :param window: (int)
    :return: (numpy array)
    """
    weights = np.repeat(1.0, window) / window
    return np.convolve(values, weights, 'valid')

def plot_results(log_folder, legend='train', N=-1):
    """
    plot the results

    :param log_folder: (str) the save location of the results to plot
    :param title: (str) the title of the task to plot
    """
    x, y = ts2xy(load_results(log_folder), 'timesteps')
    y = moving_average(y, window=100)
    # Truncate x
    x = x[len(x) - len(y):]
    if N > 0: 
        x = x[:N]
        y = y[:N]
    plt.plot(x, y, label=legend)
    plt.xlabel('Number of Timesteps')
    plt.ylabel('Rewards')

algo = 'dqn'
seed = 2

algos = ['dqn', 'dqn', 'trpo', 'trpo', 'trpo', 'trpo']
seeds = [2, 2, 1, 1, 1, 1]
tasks = ['lfd','shaping', 'gail', 'pofd', 'shaping', 'train']
legends = ['DQfD','DQ-RS', 'GAIL', 'POfD', 'TRPO-RS', 'TRPO']

#env = 'Acrobot-v1'
#tasks = ['lfd','train', 'shaping', 'shaping-decay']
#legends = ['learn-from-demonstraion','learn-from-scratch', 'learn-from-reward-shaping', 'decay-reward-shaping']

#algo = 'trpo'
#seed = 1
#env = 'CartPole-v1'
#tasks = ['train']
#legends = ['TRPO']

env = 'BreakoutNoFrameskip-v4'
env = 'QbertNoFrameskip-v4'
env = 'GravitarNoFrameskip-v4'
env = 'PongNoFrameskip-v4'
algos = ['dqn', 'dqn']
algos = ['trpo', 'trpo']
algos = ['ppo2', 'ppo2']
seeds = [2, 1]
seeds = [1, 1]
seeds = [2, 1]
#tasks = ['train']
tasks = ['shaping','train']
legends = ['Shaping','DuelDQN']
legends = ['Shaping','TRPO']
legends = ['Shaping','PPO2']

algos=['trpo']
tasks=['train']
legends=['Train']
### for shaping ppo2 on pong, 1 is default, 2 is using 52 episodes, 3 is using 10% of 499 episodes
env = 'Acrobot-v1'
env = 'CartPole-v1'
env = 'HalfCheetah-v2'
env = 'Hopper-v2'
algos = ['trpo', 'trpo', 'trpo', 'trpo']
legends = ['TRPO', 'GAIL', 'POfD', 'SWITCH']
tasks = ['train-sparse', 'gail-sparse', 'pofd-sparse', 'switch-sparse']
tasks = ['train', 'gail', 'pofd', 'switch']
ids = [0, 1, 2, 3]
seeds=[2]
ids = [0]


for seed in seeds:
    figure_name = env.split('-')[0] + '-' + '-'.join(legends[i] for i in ids) + '-seed{}'.format(seed)
    if_sparse = '-SPARSE' if 'sparse' in tasks[ids[0]] else ''
    figure_name += if_sparse 
    fig = plt.figure(figure_name)

    for i in ids:#range(len(tasks)):
        task, legend, algo = tasks[i], legends[i], algos[i]
        path = os.path.join(PATH_PREFIX, task, algo, env, "rank{}".format(seed))
        print(path)

        plot_results(path, legend=legend)

    plt.legend()
    plt.savefig('../figs/{}.pdf'.format(figure_name))
    print(figure_name)
