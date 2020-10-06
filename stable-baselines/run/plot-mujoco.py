import seaborn as sns
import argparse
import matplotlib.pyplot as plt
sns.set()
current_palette = sns.color_palette()
sns.palplot(current_palette)
import pandas as pd
import numpy as np
import os
from settings import CONFIGS


legend_loc = "upper left"
def plot_results_game(experimental_data, title, game, agent_name, num_steps_per_iteration, num_iter, savefig=False, legend_loc=legend_loc, bc_file=None):
    fig, ax = plt.subplots(figsize=(15,10))
    sns.tsplot(data=experimental_data, time='iteration', unit='run_number',
             condition='agent', value='train_episode_reward', ax=ax)
    ### plot demonstration and BC baseline
    demo_score = CONFIGS[game]['demo_score']
    x = [i for i in range(num_iter)]
    y = [demo_score for _ in x] 
    ax.plot(x, y, label='Demonstration', marker='*',color='grey')
    if bc_file is not None:
        bc_mean, bc_std = get_bc_score(game, agent_name, bc_file)
        x = [i for i in range(num_iter)]
        y = [bc_mean for _ in x] 
        ax.plot(x, y, label='BC', marker='.')


    fontsize = 22
    title_fontsize = 20
    for tick in ax.xaxis.get_major_ticks():
        tick.label1.set_fontsize(fontsize)
        tick.label1.set_fontweight('bold')
    for tick in ax.yaxis.get_major_ticks():
        tick.label1.set_fontsize(fontsize)
        tick.label1.set_fontweight('bold')

    yaxis_label, xaxis_label = "Returns", "Steps ({}K)".format(int(num_steps_per_iteration //1000))
    fontsize = "30"
    title_axis_font = {'size': title_fontsize, 'weight': 'bold'}
    xylabel_axis_font = {'size': fontsize, 'weight': 'bold'}
    ax.set_ylabel(yaxis_label, **xylabel_axis_font)
    ax.set_xlabel(xaxis_label, **xylabel_axis_font)
    ax.set_title(title, **title_axis_font)
    legend_properties = {'weight':'bold','size':"18"}
    ax.legend(loc=legend_loc, prop=legend_properties)
    ax.legend(loc=legend_loc, prop=legend_properties)

    # plt.show()
    if savefig:
        title = title.replace(" ", "") 
        plt.savefig('../figs/{}.pdf'.format(title))
        plt.savefig('../figs/{}.eps'.format(title),format='eps')
        plt.close()
    else:
        plt.show()

    print("Figure {}.pdf saved to {}".format(title, "../figs/"))
    return experimental_data

def moving_average(values, window):
    """
    Smooth values by doing a moving average
    :param values: (numpy array)
    :param window: (int)
    :return: (numpy array)
    """
    weights = np.repeat(1.0, window) / window
    return np.convolve(values, weights, 'valid')

def get_bc_score(game, agent_name, csv_file):
    df = pd.read_csv(csv_file, header=1).dropna()
    df.loc[:, 'l'] = df.loc[:, 'l'].cumsum()
    #eplen = np.array(df.loc[:, 'l'].tolist())
    eprewmean = np.array(df.loc[:, 'r'].tolist())
    mean_rew, std_rew = np.mean(eprewmean), np.std(eprewmean)
    return mean_rew, std_rew

def formalize_agent_name(agent_name):
    if 'dac' in agent_name:
        return 'DAC'
    if 'trpo' in agent_name:
        return 'GAIL + TRPO'
    if 'gail-lfd-adaptive' in agent_name:
        return 'SAIL'
    return agent_name
def read_log(game, agent_name, log_path, num_steps_per_iteration, num_iter,seeds, warmup=0):
    results = []
    for seed in seeds:
        print(seed)
        csv_file = os.path.join(log_path, 'rank{}'.format(seed), 'agent0.monitor.csv')
        df = pd.read_csv(csv_file, header=1).dropna()
        df.loc[:, 'l'] = df.loc[:, 'l'].cumsum()
        eplen = np.array(df.loc[:, 'l'].tolist())
        eprewmean = np.array(df.loc[:, 'r'].tolist())
        # construct arary
        raw_data_ = {}
        for i in range(num_iter):
            results_per_iter = []
            step_begin = int(i * num_steps_per_iteration + warmup)
            step_end = int((i+1) * num_steps_per_iteration + warmup)
            eprew_iter = eprewmean[(eplen>step_begin) & (eplen <=step_end)]
            agent_name = formalize_agent_name(agent_name)
            results_per_iter.append(agent_name)
            results_per_iter.append(game)
            results_per_iter.append(i)  #iteration
            results_per_iter.append(np.mean(eprew_iter))  # train episode reward 
            results_per_iter.append(seed)  # run number
            results.append(results_per_iter)
        
    experimental_data = pd.DataFrame.from_records(results,  columns=["agent", "game", "iteration", "train_episode_reward", "run_number"])
    return experimental_data
    

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--env', type=str, default="Hopper-v2", help='environment ID')
    parser.add_argument('--algo', type=str, default="sail", help='Algorithm')
    parser.add_argument('--shift', type=int, default=0, help='seee index shift')
    parser.add_argument('--seeds', type=int, default=3, help='number of seeds')
    parser.add_argument('--episodes', type=int, default=1, help='number of demonstration episodes')
    parser.add_argument('--warmup', type=int, default=0, help='number of warmup samples')
    parser.add_argument('--timesteps', type=int, default=int(1e6), help='Time steps')
    parser.add_argument('--steps', type=int, default=10000, help='num steps per iteration')
    parser.add_argument('--rewards', type=str, choices=['dense', 'sparse'], default='dense', help='Environment Rewards')
    parser.add_argument('--quality', type=str, choices=['near', 'sub'], default='sub', help='Environment Rewards')
    parser.add_argument('--legends', type=str, default='gail-dac-subopt,gail-lfd-BC-adaptive-subopt', help="Methods to plot, split by comma")
    parser.add_argument('--log-dir', help='Log directory', type=str, default='/tmp/logs') # required=True,

    args = parser.parse_args()


    experimental_datas = []

    timesteps = args.timesteps
    num_steps_per_iteration = args.steps
    num_iter = int(timesteps/num_steps_per_iteration)

    #env = 'Hopper-v2'
    #env = 'CartPole-v1'
    #legends = {0: 'GAIL', 1: 'POfD', 2:'TRPO', 3: 'SWITCH'}
    #algos = ['trpo','trpo', 'trpo', 'trpo']
    algo = args.algo
    env=args.env
    seeds=[i + args.shift for i in range(1, args.seeds + 1)] # dense, near-optimal
    if args.rewards == 'dense':
        setting = 'Dense-{}Opt'.format(args.quality)
    else:
        setting = 'Sparse-{}Opt'.format(args.quality)

    if 'dual' in algo:
        setting = setting + '-Dual'


    legends = [l.strip() for l in args.legends.split(',')]
    for legend in legends:
        task = legend
        algo = args.algo
        if 'bcq' in task:
            algo = 'bcq'
        if 'dac' in task:
            algo = 'dac'
        if 'trpo' in task:
            algo = 'trpo'
        path = os.path.join(args.log_dir, task, algo, env)
        print(path)
        experimental_data = read_log(env, legend, path, num_steps_per_iteration, num_iter, seeds, warmup=args.warmup)
        experimental_datas.append(experimental_data)


    short_legends = [i.replace('-LFD', '').replace('-GAIL','') for i in legends]
    figure_name = env
    all_agents_data = pd.concat(experimental_datas, axis=0)
    savefig = True
    bc_path = os.path.join(args.log_dir, 'eval-bc-episode-{}'.format(args.episodes), args.algo, env)
    bc_file = os.path.join(bc_path, 'rank1', 'agent0.monitor.csv')
    if not os.path.isfile(bc_file):
        bc_file = None
    experimental_datas = plot_results_game(all_agents_data, figure_name, env, legends, num_steps_per_iteration, num_iter, savefig, bc_file=bc_file)
