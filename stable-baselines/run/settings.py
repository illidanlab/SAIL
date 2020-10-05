CONFIGS = {
    'Humanoid-v2': {
        'optimal_score': 330,
        'max_score': 330,
        'demo_score': 330,
        'n_episodes': 10,
        'buffer_size': 800,
        'n_epochs': 5000,
        'pretrain_timesteps': int(1e4),
        'sample_fraction':1,
        'shaping_scale':1,  # only used when run task 'shaping-decay'
        'pofd_lambda1': 0.1,
        'pofd_lambda0': 1,
        'lagrange_stepsize': 1e-3,
        'min_expert_adv': 0.22
    },

    'Ant-v2': {
        'optimal_score': 1200,
        'demo_score': 1200,
        'max_score': 1200,
        'n_episodes': 18,
        'buffer_size': int(5*1e4),
        'demo_buffer_size': int(1e3),
        'n_epochs': 100,
        'sample_fraction':0.1,
        'pofd_lambda1': 0.1,
        'pofd_lambda0': 1,
        'lagrange_stepsize': 1e-3,
        'entropy_lamda': 1e-3,
        'shaping_scale': 1e-4 # initial reward shaping scale
    },

    'Reacher-v2': {
        'optimal_score': -3.6,
        'max_score': -3.75,
        'demo_score': -3.6,
        'n_episodes': 1,
        'buffer_size': int(5*1e4),
        'n_epochs': 100,
        'sample_fraction':0.1,
        'pofd_lambda1': 1,
        'pofd_lambda0': 1,
        'lagrange_stepsize': 1e-3,
        'entropy_lamda': 1e-3,
        'shaping_scale': 1e-4 # initial reward shaping scale
    },

    'Swimmer-v2': {
        'optimal_score': 120,
        'demo_score': 120,
        'max_score': 120,
        'n_episodes': 18,
        'buffer_size': int(5*1e4),
        'n_epochs': 100,
        'sample_fraction':0.1,
        'pofd_lambda1': 1,
        'pofd_lambda0': 1,
        'lagrange_stepsize': 1e-3,
        'entropy_lamda': 1e-3,
        'shaping_scale': 1e-4 # initial reward shaping scale
    },

    'HalfCheetah-v2': {
        'optimal_score': 5600,
        'max_score': 5000,
        'demo_score': 5600,
        'n_episodes': 1,
        'buffer_size': int(5*1e4),
        'demo_buffer_size': int(1e3),
        'n_epochs': 100,
        'sample_fraction':0.1,
        'pofd_lambda1': 1,
        'pofd_lambda0': 1,
        'entropy_lamda': 1e-3,
        'lagrange_stepsize': 1e-3,
        'shaping_scale': 1e-4 # initial reward shaping scale
    },

    'Hopper-v2': {
        'max_score': 1500,
        'demo_score': 1500,
        'optimal_score': 1500,
        'n_episodes': 10,
        'buffer_size': int(5*1e4),
        'demo_buffer_size': int(1e3),
        'n_epochs': 100,
        'pretrain_timesteps': int(1e4),
        'sample_fraction':1,
        'pofd_lambda1': 1,
        'pofd_lambda0': 1,
        'lagrange_stepsize': 1e-4,
        'shaping_scale': 1e-4 # initial reward shaping scale
    },

    'Walker2d-v2': {
        'optimal_score': 1500,
        'max_score': 1500,
        'demo_score': 1500,
        'n_episodes': 11,
        'buffer_size': int(5*1e4),
        'demo_buffer_size': int(1e3),
        'n_epochs': 100,
        'sample_fraction':1,
        'entropy_lamda': 1e-3,
        'pofd_lambda1': 1,
        'pofd_lambda0': 1,
        'lagrange_stepsize': 1e-4,
        'shaping_scale': 1e-4 # initial reward shaping scale
    }
}
