import sys
import time
import multiprocessing
from collections import deque
import warnings

import gym
import numpy as np
import tensorflow as tf

from stable_baselines.a2c.utils import total_episode_reward_logger
from stable_baselines.common import tf_util, OffPolicyRLModel, SetVerbosity, TensorboardWriter, fmt_row
from stable_baselines.common.vec_env import VecEnv
from stable_baselines.deepq.replay_buffer import ReplayBuffer, ReplayBufferExtend, TrajectoryBuffer
from stable_baselines.ppo2.ppo2 import safe_mean, get_schedule_fn
from stable_baselines.sac.policies import SACPolicy
from stable_baselines import logger
from stable_baselines.gail.adversary import DiscriminatorCalssifier
from stable_baselines.gail.dataset.dataset import ExpertDataset

MIN_P = 1e-2
MAX_P = 1 - 1e-2

def get_vars(scope):
    """
    Alias for get_trainable_vars

    :param scope: (str)
    :return: [tf Variable]
    """
    return tf_util.get_trainable_vars(scope)


class SAC(OffPolicyRLModel):
    """
    Soft Actor-Critic (SAC)
    Off-Policy Maximum Entropy Deep Reinforcement Learning with a Stochastic Actor,
    This implementation borrows code from original implementation (https://github.com/haarnoja/sac)
    from OpenAI Spinning Up (https://github.com/openai/spinningup) and from the Softlearning repo
    (https://github.com/rail-berkeley/softlearning/)
    Paper: https://arxiv.org/abs/1801.01290
    Introduction to SAC: https://spinningup.openai.com/en/latest/algorithms/sac.html

    :param policy: (SACPolicy or str) The policy model to use (MlpPolicy, CnnPolicy, LnMlpPolicy, ...)
    :param env: (Gym environment or str) The environment to learn from (if registered in Gym, can be str)
    :param gamma: (float) the discount factor
    :param learning_rate: (float or callable) learning rate for adam optimizer,
        the same learning rate will be used for all networks (Q-Values, Actor and Value function)
        it can be a function of the current progress (from 1 to 0)
    :param buffer_size: (int) size of the replay buffer
    :param batch_size: (int) Minibatch size for each gradient update
    :param tau: (float) the soft update coefficient ("polyak update", between 0 and 1)
    :param ent_coef: (str or float) Entropy regularization coefficient. (Equivalent to
        inverse of reward scale in the original SAC paper.)  Controlling exploration/exploitation trade-off.
        Set it to 'auto' to learn it automatically (and 'auto_0.1' for using 0.1 as initial value)
    :param train_freq: (int) Update the model every `train_freq` steps.
    :param learning_starts: (int) how many steps of the model to collect transitions for before learning starts
    :param target_update_interval: (int) update the target network every `target_network_update_freq` steps.
    :param gradient_steps: (int) How many gradient update after each step
    :param target_entropy: (str or float) target entropy when learning ent_coef (ent_coef = 'auto')
    :param action_noise: (ActionNoise) the action noise type (None by default), this can help
        for hard exploration problem. Cf DDPG for the different action noise type.
    :param random_exploration: (float) Probability of taking a random action (as in an epsilon-greedy strategy)
        This is not needed for SAC normally but can help exploring when using HER + SAC.
        This hack was present in the original OpenAI Baselines repo (DDPG + HER)
    :param verbose: (int) the verbosity level: 0 none, 1 training information, 2 tensorflow debug
    :param tensorboard_log: (str) the log location for tensorboard (if None, no logging)
    :param _init_setup_model: (bool) Whether or not to build the network at the creation of the instance
    :param policy_kwargs: (dict) additional arguments to be passed to the policy on creation
    :param full_tensorboard_log: (bool) enable additional logging when using tensorboard
        Note: this has no effect on SAC logging for now
    """

    def __init__(self, policy, env, gamma=0.99, learning_rate=3e-4, buffer_size=50000,
                 learning_starts=100, train_freq=1, batch_size=64,
                 tau=0.005, ent_coef='auto', target_update_interval=1,
                 gradient_steps=1, target_entropy='auto', action_noise=None,
                 random_exploration=0.0, verbose=0, tensorboard_log=None,
                 _init_setup_model=True, policy_kwargs=None, full_tensorboard_log=False, 
                 using_gail=False, lfd=False, adaptive=False, shaping_mode=None, huber_loss=False,
                 expert_data_path=None, n_jobs=1):

        super(SAC, self).__init__(policy=policy, env=env, replay_buffer=None, verbose=verbose,
                                  policy_base=SACPolicy, requires_vec_env=False, policy_kwargs=policy_kwargs)

        self.buffer_size = buffer_size
        self.learning_rate = learning_rate
        self.learning_starts = learning_starts
        self.train_freq = train_freq 
        self.batch_size = batch_size
        self.tau = tau
        # In the original paper, same learning rate is used for all networks
        # self.policy_lr = learning_rate
        # self.qf_lr = learning_rate
        # self.vf_lr = learning_rate
        # Entropy coefficient / Entropy temperature
        # Inverse of the reward scale
        self.ent_coef = ent_coef
        self.target_update_interval = target_update_interval
        self.gradient_steps = gradient_steps
        self.gamma = gamma
        self.action_noise = action_noise
        self.random_exploration = random_exploration

        self.value_fn = None
        self.graph = None
        self.replay_buffer = None
        self.episode_reward = None
        self.sess = None
        self.tensorboard_log = tensorboard_log
        self.verbose = verbose
        self.params = None
        self.summary = None
        self.policy_tf = None
        self.target_entropy = target_entropy
        self.full_tensorboard_log = full_tensorboard_log

        self.obs_target = None
        self.target_policy = None
        self.actions_ph = None
        self.rewards_ph = None
        self.terminals_ph = None
        self.observations_ph = None
        self.action_target = None
        self.next_observations_ph = None
        self.value_target = None
        self.step_ops = None
        self.target_update_op = None
        self.infos_names = None
        self.entropy = None
        self.target_params = None
        self.learning_rate_ph = None
        self.processed_obs_ph = None
        self.processed_next_obs_ph = None
        self.log_ent_coef = None

        self.using_gail = using_gail
        self.lfd = lfd
        self.huber_loss = huber_loss
        self.adaptive = adaptive
        self.n_jobs = n_jobs
        self.demo_buffer_size = 1e4
        self.expert_scores = [] 
        self.episode_buffer = TrajectoryBuffer(1e5,gamma=gamma) 
        self.d_gradient_steps=10
        self.train_discriminator_freq=500
        self.expert_data_path = expert_data_path
        
        if self.using_gail or self.lfd:
            self.expert_dataset = ExpertDataset(expert_path=self.expert_data_path, ob_flatten=False)
            print('-'*20 + "expert_data_path: {}".format(self.expert_data_path))
            time.sleep(4)
            n_samples = len(self.expert_dataset.observations) 

        if _init_setup_model:
            self.setup_model()

    def _get_pretrain_placeholders(self):
        policy = self.policy_tf
        # Rescale
        deterministic_action = self.deterministic_action * np.abs(self.action_space.low)
        return policy.obs_ph, self.actions_ph, deterministic_action

    def setup_model(self):
        with SetVerbosity(self.verbose):
            self.graph = tf.Graph()
            with self.graph.as_default():
                n_cpu = multiprocessing.cpu_count()
                if sys.platform == 'darwin':
                    n_cpu //= 2
                #self.sess = tf_util.make_session(num_cpu=n_cpu, graph=self.graph)
                cpu_slice_num = max(1, n_cpu // self.n_jobs)
                print("Total CPU:{}, CPU each job: {}".format(n_cpu, cpu_slice_num))
                self.sess = tf_util.make_session(num_cpu=cpu_slice_num, graph=self.graph)

                #self.replay_buffer = ReplayBuffer(self.buffer_size)
                if self.using_gail:
                    self.discriminator = DiscriminatorCalssifier( 
                        self.observation_space,
                        self.action_space,
                        256,
                        entcoeff=0.01,
                        gradcoeff=10)

                if self.using_gail or self.lfd:
                    # initialize R_E: demonstration-buffer and R: self-generated data buffer
                    self.replay_buffer = ReplayBufferExtend(self.buffer_size)
                    self.demo_replay_buffer = ReplayBufferExtend(self.demo_buffer_size)
                else:
                    # initialize R: self-generated data buffer
                    self.replay_buffer = ReplayBufferExtend(self.buffer_size)


                with tf.variable_scope("input", reuse=False):
                    # Create policy and target TF objects
                    self.policy_tf = self.policy(self.sess, self.observation_space, self.action_space,
                                                 **self.policy_kwargs)
                    self.target_policy = self.policy(self.sess, self.observation_space, self.action_space,
                                                     **self.policy_kwargs)

                    # Initialize Placeholders
                    self.observations_ph = self.policy_tf.obs_ph
                    # Normalized observation for pixels
                    self.processed_obs_ph = self.policy_tf.processed_obs
                    self.next_observations_ph = self.target_policy.obs_ph
                    self.processed_next_obs_ph = self.target_policy.processed_obs
                    self.action_target = self.target_policy.action_ph
                    self.terminals_ph = tf.placeholder(tf.float32, shape=(None, 1), name='terminals')
                    self.rewards_ph = tf.placeholder(tf.float32, shape=(None, 1), name='rewards')
                    self.actions_ph = tf.placeholder(tf.float32, shape=(None,) + self.action_space.shape,
                                                     name='actions')
                    self.learning_rate_ph = tf.placeholder(tf.float32, [], name="learning_rate_ph")
                    self.d_learning_rate_ph = tf.placeholder(tf.float32, [], name="d_learning_rate_ph")

                with tf.variable_scope("model", reuse=False):
                    # Create the policy
                    # first return value corresponds to deterministic actions
                    # policy_out corresponds to stochastic actions, used for training
                    # logp_pi is the log probabilty of actions taken by the policy
                    self.deterministic_action, policy_out, logp_pi = self.policy_tf.make_actor(self.processed_obs_ph)
                    # Monitor the entropy of the policy,
                    # this is not used for training
                    self.entropy = tf.reduce_mean(self.policy_tf.entropy)
                    #  Use two Q-functions to improve performance by reducing overestimation bias.
                    qf1, qf2, value_fn = self.policy_tf.make_critics(self.processed_obs_ph, self.actions_ph,
                                                                     create_qf=True, create_vf=True)
                    qf1_pi, qf2_pi, _ = self.policy_tf.make_critics(self.processed_obs_ph,
                                                                    policy_out, create_qf=True, create_vf=False,
                                                                    reuse=True)

                    # Target entropy is used when learning the entropy coefficient
                    if self.target_entropy == 'auto':
                        # automatically set target entropy if needed
                        self.target_entropy = -np.prod(self.env.action_space.shape).astype(np.float32)
                    else:
                        # Force conversion
                        # this will also throw an error for unexpected string
                        self.target_entropy = float(self.target_entropy)

                    # The entropy coefficient or entropy can be learned automatically
                    # see Automating Entropy Adjustment for Maximum Entropy RL section
                    # of https://arxiv.org/abs/1812.05905
                    if isinstance(self.ent_coef, str) and self.ent_coef.startswith('auto'):
                        # Default initial value of ent_coef when learned
                        init_value = 1.0
                        if '_' in self.ent_coef:
                            init_value = float(self.ent_coef.split('_')[1])
                            assert init_value > 0., "The initial value of ent_coef must be greater than 0"

                        self.log_ent_coef = tf.get_variable('log_ent_coef', dtype=tf.float32,
                                                            initializer=np.log(init_value).astype(np.float32))
                        #self.clipped_log_ent_coef = tf.clip_by_value(self.log_ent_coef, -3, 3)
                        self.ent_coef = tf.exp(self.log_ent_coef)
                    else:
                        # Force conversion to float
                        # this will throw an error if a malformed string (different from 'auto')
                        # is passed
                        self.ent_coef = float(self.ent_coef)


                with tf.variable_scope("target", reuse=False):
                    # Create the value network
                    _, _, value_target = self.target_policy.make_critics(self.processed_next_obs_ph,
                                                                         create_qf=False, create_vf=True)
                    self.value_target = value_target

                if self.using_gail:
                    with tf.variable_scope("discriminator_loss", reuse=False):
                        gail_optimizer = tf.train.AdamOptimizer(learning_rate=self.d_learning_rate_ph)
                        self.gail_item_loss = self.discriminator.losses
                        self.gail_total_loss = self.discriminator.total_loss
                        self.gail_sample_loss = self.discriminator.sample_loss
                        self.gail_train_op = gail_optimizer.minimize(self.gail_total_loss,  var_list=self.discriminator.get_trainable_variables())
                        # Log discriminator scalars for debugging purposes
                        gail_scalar_summaries = []
                        for i, loss_name in enumerate(self.discriminator.loss_name):
                            i = tf.summary.scalar(loss_name, self.gail_item_loss[i])
                            gail_scalar_summaries.append(i)
                        self.gail_summary = tf.summary.merge(gail_scalar_summaries)

                with tf.variable_scope("loss", reuse=False):
                    # Take the min of the two Q-Values (Double-Q Learning)
                    min_qf_pi = tf.minimum(qf1_pi, qf2_pi)

                    # Target for Q value regression
                    q_backup = tf.stop_gradient(
                        self.rewards_ph +
                        (1 - self.terminals_ph) * self.gamma * self.value_target
                    )

                    # Compute Q-Function loss
                    # TODO: test with huber loss (it would avoid too high values)
                    if self.huber_loss:
                        qf1_loss = tf.losses.huber_loss(q_backup, qf1, delta=1.0) 
                        qf2_loss = tf.losses.huber_loss(q_backup, qf2, delta=1.0) 
                    else:
                        qf1_loss = 0.5 * tf.reduce_mean((q_backup - qf1) ** 2)
                        qf2_loss = 0.5 * tf.reduce_mean((q_backup - qf2) ** 2)

                    # Compute the entropy temperature loss
                    # it is used when the entropy coefficient is learned
                    ent_coef_loss, entropy_optimizer = None, None
                    if not isinstance(self.ent_coef, float):
                        ent_coef_loss = -tf.reduce_mean(
                            self.log_ent_coef * tf.stop_gradient(logp_pi + self.target_entropy))
                        entropy_optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate_ph)

                    # Compute the policy loss
                    # Alternative: policy_kl_loss = tf.reduce_mean(logp_pi - min_qf_pi)
                    policy_kl_loss = tf.reduce_mean(self.ent_coef * logp_pi - min_qf_pi)

                    # NOTE: in the original implementation, they have an additional
                    # regularization loss for the gaussian parameters
                    # this is not used for now
                    # policy_loss = (policy_kl_loss + policy_regularization_loss)
                    policy_loss = policy_kl_loss


                    # Target for value fn regression
                    # We update the vf towards the min of two Q-functions in order to
                    # reduce overestimation bias from function approximation error.
                    v_backup = tf.stop_gradient(min_qf_pi - self.ent_coef * logp_pi)
                    value_loss = 0.5 * tf.reduce_mean((value_fn - v_backup) ** 2)

                    values_losses = qf1_loss + qf2_loss + value_loss

                    # Policy train op
                    # (has to be separate from value train op, because min_qf_pi appears in policy_loss)
                    policy_optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate_ph)
                    self.policy_train_op = policy_train_op = policy_optimizer.minimize(policy_loss, var_list=get_vars('model/pi'))

                    # Value train op
                    value_optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate_ph)
                    values_params = get_vars('model/values_fn')

                    source_params = get_vars("model/values_fn/vf")
                    target_params = get_vars("target/values_fn/vf")

                    # Polyak averaging for target variables
                    self.target_update_op = [
                        tf.assign(target, (1 - self.tau) * target + self.tau * source)
                        for target, source in zip(target_params, source_params)
                    ]
                    # Initializing target to match source variables
                    target_init_op = [
                        tf.assign(target, source)
                        for target, source in zip(target_params, source_params)
                    ]

                    # Control flow is used because sess.run otherwise evaluates in nondeterministic order
                    # and we first need to compute the policy action before computing q values losses
                    with tf.control_dependencies([policy_train_op]):
                        train_values_op = value_optimizer.minimize(values_losses, var_list=values_params)

                        self.infos_names = ['policy_loss', 'qf1_loss', 'qf2_loss', 'value_loss', 'entropy']
                        # All ops to call during one training step
                        self.step_ops = [policy_loss, qf1_loss, qf2_loss,
                                         value_loss, qf1, qf2, value_fn, logp_pi,
                                         self.entropy, policy_train_op, train_values_op]

                        # Add entropy coefficient optimization operation if needed
                        if ent_coef_loss is not None:
                            with tf.control_dependencies([train_values_op]):
                                ent_coef_op = entropy_optimizer.minimize(ent_coef_loss, var_list=self.log_ent_coef)
                                self.infos_names += ['ent_coef_loss', 'ent_coef']
                                self.step_ops += [ent_coef_op, ent_coef_loss, self.ent_coef]

                    # Monitor losses and entropy in tensorboard
                    train_summary = []
                    i = tf.summary.scalar('policy_loss', policy_loss)
                    train_summary.append(i)
                    i = tf.summary.scalar('qf1_loss', qf1_loss)
                    train_summary.append(i)
                    i = tf.summary.scalar('qf2_loss', qf2_loss)
                    train_summary.append(i)
                    i = tf.summary.scalar('value_loss', value_loss)
                    train_summary.append(i)
                    i = tf.summary.scalar('entropy', self.entropy)
                    train_summary.append(i)
                    if ent_coef_loss is not None:
                        i = tf.summary.scalar('ent_coef_loss', ent_coef_loss)
                        train_summary.append(i)
                        i = tf.summary.scalar('ent_coef', self.ent_coef)
                        train_summary.append(i)

                    i = tf.summary.scalar('learning_rate', tf.reduce_mean(self.learning_rate_ph))
                    train_summary.append(i)

                # Retrieve parameters that must be saved
                self.params = get_vars("model")
                self.target_params = get_vars("target/values_fn/vf")

                # Initialize Variables and target network
                with self.sess.as_default():
                    self.sess.run(tf.global_variables_initializer())
                    self.sess.run(target_init_op)

                self.summary = tf.summary.merge(train_summary)


    def generate_train_data(self, mode=None):
        # Sample a batch from the replay buffer
        expert_batch_size = self_batch_size = self.batch_size // 2
        #self_batch_size = int(self.batch_size * 0.9)
        #expert_batch_size = int(self.batch_size * 0.1)

        if self.lfd: # sample both self-data and expert data
            expert_batch = self.demo_replay_buffer.sample(expert_batch_size)
            self_batch = self.replay_buffer.sample(self_batch_size)
            batch = [np.concatenate((expert_batch[i],self_batch[i]),axis=0) for i in range(len(self_batch))] # numpy array
        else:
            batch = self.replay_buffer.sample(self.batch_size)
        return batch


    def generate_discriminator_data(self, mode = None):
        expert_batch = self.demo_replay_buffer.sample(self.batch_size )
        ob_expert, ac_expert = expert_batch[:2]
        # disturb expert actions if needed
        #ac_expert = self.shift_actions(ac_expert)

        batch = self.replay_buffer.sample(self.batch_size )
        ob_batch, ac_batch =  batch[:2]
        ob_expert, ac_expert = expert_batch[:2]

        # update running mean/std for discriminator
        if self.discriminator.normalize: 
            self.discriminator.obs_rms.update(np.concatenate((ob_batch, ob_expert), 0),sess=self.sess) 

        # Reshape actions if needed when using discrete actions
        if isinstance(self.action_space, gym.spaces.Discrete):
            if len(ac_batch.shape) == 2:
                ac_batch = ac_batch[:, 0]
            if len(ac_expert.shape) == 2:
                ac_expert = ac_expert[:, 0]

        return ob_expert, ac_expert, ob_batch, ac_batch 

    def _train_discriminator(self, writer, logger, step, current_d_lr, shaping_mode):
        
        #timesteps_per_batch = len(observation)  
        if step % 1000 == 0:
            logger.log("Optimizing Discriminator...")
            logger.log(fmt_row(13, self.discriminator.loss_name + ['discriminator-total-loss'] ))

        # NOTE: uses only the last g step for observation
        d_losses = []  # list of tuples, each of which gives the loss for a minibatch
        for i in range(self.d_gradient_steps):
            ob_expert, ac_expert, ob_batch, ac_batch = self.generate_discriminator_data(shaping_mode)
            #print(ob_batch.shape)

            feed_dict = {
                self.discriminator.generator_obs_ph: ob_batch, 
                self.discriminator.generator_acs_ph: ac_batch,
                self.discriminator.expert_obs_ph: ob_expert,
                self.discriminator.expert_acs_ph: ac_expert,
                self.d_learning_rate_ph: current_d_lr
            }

            if writer is not None:
                run_ops = [self.gail_summary] + self.gail_sample_loss + self.gail_item_loss + [self.gail_total_loss, self.gail_train_op]
                out = self.sess.run(run_ops, feed_dict)
                summary = out.pop(0)
                writer.add_summary(summary, step)
            else:
                run_ops = self.gail_sample_loss + self.gail_item_loss + [self.gail_total_loss, self.gail_train_op]
                out = self.sess.run(run_ops, feed_dict)

            losses = out[2:-1]
            d_losses.append(losses)
            del ob_batch, ac_batch, ob_expert, ac_expert 

        logger.log(fmt_row(13, np.mean(d_losses, axis=0)))


    def get_imitate_reward(self, observation, action, true_reward):
        _, p = self.discriminator.get_confidence(observation, action, sess=self.sess)
        mode = self.config['shaping_mode']
        if 'airl' in mode:
            p = np.clip(p, MIN_P, MAX_P) 
            r =  np.log(p) - np.log(1 - p) - np.log(MIN_P) + np.log(MAX_P) # constant shift
        elif 'bound' in mode:
            left_p = np.clip(p, 0, 0.5)
            right_p = np.clip(p, 0.5, 1)
            bonus_left = np.log(0.5 - left_p + 1e-8) * (0.5-left_p) + np.log(left_p + 0.5)*(left_p + 0.5) + np.log(2)
            bonus_right = -np.log(1-right_p + 0.5 + 1e-8) * (1-right_p + 0.5) - np.log(right_p - 0.5 + 1e-8) * (right_p - 0.5) + np.log(2) 
            left_idx = p <= 0.5
            right_idx = p > 0.5 
            r = left_idx * bonus_left + right_idx * bonus_right 
        else:
            p = np.clip(p, MIN_P, MAX_P) 
            #r =  np.log(p) - np.log(1 - p) - np.log(MIN_P) + np.log(MAX_P) # constant shift
            r =  - np.log(1 - p) 
        return r

    def _train_step(self, step, writer, learning_rate):
        # Sample a batch from the replay buffer
        #batch = self.replay_buffer.sample(self.batch_size)
        batch = self.generate_train_data()
        batch_obs, batch_actions, batch_rewards, batch_next_obs, batch_dones = batch[:5]
        print(batch_rewards.shape)

        if self.using_gail:
            batch_rewards = self.get_imitate_reward(batch_obs, batch_actions, batch_rewards)

        batch_rewards = np.array(batch_rewards).reshape(-1,1) 
        #print(batch_rewards.shape)

        feed_dict = {
            self.observations_ph: batch_obs,
            self.actions_ph: batch_actions,
            self.next_observations_ph: batch_next_obs,
            self.rewards_ph: batch_rewards.reshape(self.batch_size, -1),
            self.terminals_ph: batch_dones.reshape(self.batch_size, -1),
            self.learning_rate_ph: learning_rate
        }

        # out  = [policy_loss, qf1_loss, qf2_loss,
        #         value_loss, qf1, qf2, value_fn, logp_pi,
        #         self.entropy, policy_train_op, train_values_op]

        # Do one gradient step
        # and optionally compute log for tensorboard
        if writer is not None:
            out = self.sess.run([self.summary] + self.step_ops, feed_dict)
            summary = out.pop(0)
            writer.add_summary(summary, step)
        else:
            out = self.sess.run(self.step_ops, feed_dict)

        # Unpack to monitor losses and entropy
        policy_loss, qf1_loss, qf2_loss, value_loss, *values = out
        # qf1, qf2, value_fn, logp_pi, entropy, *_ = values
        entropy = values[4]

        if self.log_ent_coef is not None:
            ent_coef_loss, ent_coef = values[-2:]
            return policy_loss, qf1_loss, qf2_loss, value_loss, entropy, ent_coef_loss, ent_coef

        return policy_loss, qf1_loss, qf2_loss, value_loss, entropy


    def initialize_expert_buffer(self):
        demo_obs, demo_actions, demo_rewards, demo_dones, demo_next_obs, demo_episode_scores = self.expert_dataset.get_transitions()   
        episode_lengths = np.where(demo_dones == 1)[0]
        n_samples = len(demo_obs)
        # get episode_score for each demo sample, either 0 or episode-reward
        episode_idx, n_episodes = 0, len(demo_episode_scores)
        for idx in range(n_samples-1):
            episode_score = demo_episode_scores[episode_idx] 
            episode_length = episode_lengths[episode_idx] 
            true_reward = demo_rewards[idx]
            if demo_dones[idx+1] == 1:
                print(idx, 'episode_score for demonstration tarjectory: {}'.format(episode_score)) 
                episode_idx += 1 
                self.expert_scores.append(episode_score)
                assert episode_length - idx >= 0
            
            self.demo_replay_buffer.add(demo_obs[idx], demo_actions[idx], demo_rewards[idx], demo_obs[idx + 1], float(demo_dones[idx]), 1.0, 0, true_reward)
            if idx % 1000 == 0:
                print("Adding demonstration to the replay buffer, processing {} %  ..".format(float(idx+1) * 100 / n_samples))
        true_reward = demo_rewards[-1]
        self.demo_replay_buffer.add(demo_obs[-1], demo_actions[-1], demo_rewards[-1], demo_obs[-1], float(demo_dones[-1]), 1.0, 0, true_reward)
        self.expert_scores.sort()

    def learn(self, total_timesteps, callback=None, seed=None,
              log_interval=4, tb_log_name="SAC", reset_num_timesteps=True, replay_wrapper=None, config=None):

        new_tb_log = self._init_num_timesteps(reset_num_timesteps)
        self.config = config
        shaping_mode = self.config['shaping_mode']

        if replay_wrapper is not None:
            self.replay_buffer = replay_wrapper(self.replay_buffer)

        if self.using_gail or self.lfd:
            self.initialize_expert_buffer()

        with SetVerbosity(self.verbose), TensorboardWriter(self.graph, self.tensorboard_log, tb_log_name, new_tb_log) \
                as writer:

            self._setup_learn(seed)

            # Transform to callable if needed
            self.learning_rate = get_schedule_fn(self.learning_rate)
            # Initial learning rate
            current_lr = self.learning_rate(1)

            start_time = time.time()
            episode_rewards = [0.0]
            episode_successes = []
            if self.action_noise is not None:
                self.action_noise.reset()
            obs = self.env.reset()
            self.episode_reward = np.zeros((1,))
            ep_info_buf = deque(maxlen=100)
            n_updates = 0
            infos_values = []

            for step in range(total_timesteps):
                if callback is not None:
                    # Only stop training if return value is False, not when it is None. This is for backwards
                    # compatibility with callbacks that have no return statement.
                    if callback(locals(), globals()) is False:
                        break

                # Before training starts, randomly sample actions
                # from a uniform distribution for better exploration.
                # Afterwards, use the learned policy
                # if random_exploration is set to 0 (normal setting)
                if (self.num_timesteps < self.learning_starts
                    or np.random.rand() < self.random_exploration):
                    # No need to rescale when sampling random action
                    rescaled_action = action = self.env.action_space.sample()
                else:
                    action = self.policy_tf.step(obs[None], deterministic=False).flatten()
                    # Add noise to the action (improve exploration,
                    # not needed in general)
                    if self.action_noise is not None:
                        action = np.clip(action + self.action_noise(), -1, 1)
                    # Rescale from [-1, 1] to the correct bounds
                    rescaled_action = action * np.abs(self.action_space.low)

                assert action.shape == self.env.action_space.shape

                new_obs, reward, done, info = self.env.step(rescaled_action)
                true_reward = reward

                # Store transition in the replay buffer.
                self.replay_buffer.add(obs, action, reward, new_obs, float(done),0, 0, true_reward)
                self.episode_buffer.add(obs, action, reward, new_obs, float(done), None, 0, true_reward)
                #if 'train' in config['shaping_mode']:
                #    self.replay_buffer.add(obs, action, reward, new_obs, float(done),0, 0, true_reward)
                #else: 
                #    self.episode_buffer.add(obs, action, reward, new_obs, float(done), None, 0, true_reward)
                obs = new_obs

                # Retrieve reward and episode length if using Monitor wrapper
                maybe_ep_info = info.get('episode')
                episode_score = None if maybe_ep_info is None else maybe_ep_info['r']
                if maybe_ep_info is not None:
                    ep_info_buf.extend([maybe_ep_info])
                if episode_score is not None and 'train' not in config['shaping_mode']:
                    # update two buffer 
                    if self.adaptive:
                        if episode_score > self.expert_scores[0]:
                            print("Adding new trajectory with score {} to replay buffer, expert-score {} ".format(episode_score,self.expert_scores[0]))
                            for transition in self.episode_buffer.get_episode():
                                s, a, r, s1, if_done, _, _, true_r  = transition 
                                self.demo_replay_buffer.add(s, a, r, s1, if_done, 1.0, 0, true_r)

                            self.expert_scores.pop(0)
                            self.expert_scores.append(episode_score)
                            self.expert_scores.sort()
                        #else:
                        #    for transition in self.episode_buffer.get_episode():
                        #        s, a, r, s1, if_done, _, _, true_r = transition
                        #        self.replay_buffer.add(s, a, r, s1, if_done, 0, 0, true_r)
                    #else:
                    #    for transition in self.episode_buffer.get_episode():
                    #        s, a, r, s1, if_done, _, _, true_r = transition
                    #        self.replay_buffer.add(s, a, r, s1, if_done, 0, 0, true_r)
                    #    # reset episode buffer
                    self.episode_buffer.reset()

                if writer is not None:
                    # Write reward per episode to tensorboard
                    ep_reward = np.array([reward]).reshape((1, -1))
                    ep_done = np.array([done]).reshape((1, -1))
                    self.episode_reward = total_episode_reward_logger(self.episode_reward, ep_reward,
                                                                      ep_done, writer, self.num_timesteps)

                if step % self.train_freq == 0:
                    mb_infos_vals = []
                    # Update policy, critics and target networks
                    for grad_step in range(self.gradient_steps):
                        # Break if the warmup phase is not over
                        # or if there are not enough samples in the replay buffer
                        if not self.replay_buffer.can_sample(self.batch_size) \
                           or self.num_timesteps < self.learning_starts:
                            break
                        n_updates += 1
                        # Compute current learning_rate
                        frac = 1.0 - step / total_timesteps
                        current_lr = self.learning_rate(frac)
                        # Update policy and critics (q functions)
                        mb_infos_vals.append(self._train_step(step, writer, current_lr))
                        # Update target network
                        if (step + grad_step) % self.target_update_interval == 0:
                            # Update target network
                            self.sess.run(self.target_update_op)
                    # Log losses and entropy, useful for monitor training
                    if len(mb_infos_vals) > 0:
                        infos_values = np.mean(mb_infos_vals, axis=0)

                
                if self.using_gail and step % self.train_discriminator_freq == 0: 
                    frac = 1.0 - step / total_timesteps
                    current_d_lr = self.learning_rate(frac)
                    if self.num_timesteps >= self.learning_starts:
                        self._train_discriminator(writer, logger, step, current_d_lr, shaping_mode)

                episode_rewards[-1] += reward
                if done:
                    if self.action_noise is not None:
                        self.action_noise.reset()
                    if not isinstance(self.env, VecEnv):
                        obs = self.env.reset()
                    episode_rewards.append(0.0)

                    maybe_is_success = info.get('is_success')
                    if maybe_is_success is not None:
                        episode_successes.append(float(maybe_is_success))

                if len(episode_rewards[-101:-1]) == 0:
                    mean_reward = -np.inf
                else:
                    mean_reward = round(float(np.mean(episode_rewards[-101:-1])), 1)

                num_episodes = len(episode_rewards)
                self.num_timesteps += 1
                # Display training infos
                if self.verbose >= 1 and done and log_interval is not None and len(episode_rewards) % log_interval == 0:
                    fps = int(step / (time.time() - start_time))
                    logger.logkv("episodes", num_episodes)
                    logger.logkv("mean 100 episode reward", mean_reward)
                    if len(ep_info_buf) > 0 and len(ep_info_buf[0]) > 0:
                        logger.logkv('ep_rewmean', safe_mean([ep_info['r'] for ep_info in ep_info_buf]))
                        logger.logkv('eplenmean', safe_mean([ep_info['l'] for ep_info in ep_info_buf]))
                    logger.logkv("n_updates", n_updates)
                    logger.logkv("current_lr", current_lr)
                    logger.logkv("fps", fps)
                    logger.logkv('time_elapsed', int(time.time() - start_time))
                    if len(episode_successes) > 0:
                        logger.logkv("success rate", np.mean(episode_successes[-100:]))
                    if len(infos_values) > 0:
                        for (name, val) in zip(self.infos_names, infos_values):
                            logger.logkv(name, val)
                    logger.logkv("total timesteps", self.num_timesteps)
                    logger.dumpkvs()
                    # Reset infos:
                    infos_values = []
            return self

    def action_probability(self, observation, state=None, mask=None, actions=None, logp=False):
        if actions is not None:
            raise ValueError("Error: SAC does not have action probabilities.")

        warnings.warn("Even though SAC has a Gaussian policy, it cannot return a distribution as it "
                      "is squashed by a tanh before being scaled and ouputed.")

        return None

    def predict(self, observation, state=None, mask=None, deterministic=True):
        observation = np.array(observation)
        vectorized_env = self._is_vectorized_observation(observation, self.observation_space)

        observation = observation.reshape((-1,) + self.observation_space.shape)
        actions = self.policy_tf.step(observation, deterministic=deterministic)
        actions = actions.reshape((-1,) + self.action_space.shape)  # reshape to the correct action shape
        actions = actions * np.abs(self.action_space.low)  # scale the output for the prediction

        if not vectorized_env:
            actions = actions[0]

        return actions, None

    def get_parameter_list(self):
        return (self.params +
                self.target_params)

    def save(self, save_path, cloudpickle=False):
        data = {
            "learning_rate": self.learning_rate,
            "buffer_size": self.buffer_size,
            "learning_starts": self.learning_starts,
            "train_freq": self.train_freq,
            "batch_size": self.batch_size,
            "tau": self.tau,
            "ent_coef": self.ent_coef if isinstance(self.ent_coef, float) else 'auto',
            "target_entropy": self.target_entropy,
            # Should we also store the replay buffer?
            # this may lead to high memory usage
            # with all transition inside
            # "replay_buffer": self.replay_buffer
            "gamma": self.gamma,
            "verbose": self.verbose,
            "observation_space": self.observation_space,
            "action_space": self.action_space,
            "policy": self.policy,
            "n_envs": self.n_envs,
            "action_noise": self.action_noise,
            "random_exploration": self.random_exploration,
            "_vectorize_action": self._vectorize_action,
            "policy_kwargs": self.policy_kwargs
        }

        params_to_save = self.get_parameters()

        self._save_to_file(save_path, data=data, params=params_to_save, cloudpickle=cloudpickle)
