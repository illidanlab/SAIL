import sys
import time
import multiprocessing
from collections import deque
import warnings
import random 
import numpy as np
import gym
import time
import tensorflow as tf
from stable_baselines.common.mpi_running_mean_std import RunningMeanStd
from stable_baselines.a2c.utils import total_episode_reward_logger, total_episode_logit_reward_logger
from stable_baselines.common import tf_util, OffPolicyRLModel, SetVerbosity, TensorboardWriter, fmt_row, dataset
from stable_baselines.common.schedules import LinearSchedule
from stable_baselines.common.vec_env import VecEnv
from stable_baselines.deepq.replay_buffer import TrajectoryBuffer, ReplayBuffer, ReplayBufferExtend, ExpertBuffer, PrioritizedReplayBufferExtend
from stable_baselines.ppo2.ppo2 import safe_mean, get_schedule_fn
from stable_baselines.sac.sac import get_vars
from stable_baselines.td3.policies import TD3Policy
from stable_baselines import logger
from stable_baselines.gail.adversary import DiscriminatorCalssifier, TransitionCuriosityClassifier, RandomCuriosityClassifier 
from stable_baselines.gail.vae import VAE
from stable_baselines.gail.dataset.dataset import ExpertDataset


class BCQ(OffPolicyRLModel):
    """
    Bacth Cloning Q 

    :param policy: (TD3Policy or str) The policy model to use (MlpPolicy, CnnPolicy, LnMlpPolicy, ...)
    :param env: (Gym environment or str) The environment to learn from (if registered in Gym, can be str)
    :param gamma: (float) the discount factor
    :param learning_rate: (float or callable) learning rate for adam optimizer,
        the same learning rate will be used for all networks (Q-Values and Actor networks)
        it can be a function of the current progress (from 1 to 0)
    :param buffer_size: (int) size of the replay buffer
    :param batch_size: (int) Minibatch size for each gradient update
    :param tau: (float) the soft update coefficient ("polyak update" of the target networks, between 0 and 1)
    :param policy_delay: (int) Policy and target networks will only be updated once every policy_delay steps
        per training steps. The Q values will be updated policy_delay more often (update every training step).
    :param action_noise: (ActionNoise) the action noise type. Cf DDPG for the different action noise type.
    :param target_policy_noise: (float) Standard deviation of gaussian noise added to target policy
        (smoothing noise)
    :param target_noise_clip: (float) Limit for absolute value of target policy smoothing noise.
    :param train_freq: (int) Update the model every `train_freq` steps.
    :param learning_starts: (int) how many steps of the model to collect transitions for before learning starts
    :param gradient_steps: (int) How many gradient update after each step
    :param random_exploration: (float) Probability of taking a random action (as in an epsilon-greedy strategy)
        This is not needed for TD3 normally but can help exploring when using HER + TD3.
        This hack was present in the original OpenAI Baselines repo (DDPG + HER)
    :param verbose: (int) the verbosity level: 0 none, 1 training information, 2 tensorflow debug
    :param tensorboard_log: (str) the log location for tensorboard (if None, no logging)
    :param _init_setup_model: (bool) Whether or not to build the network at the creation of the instance
    :param policy_kwargs: (dict) additional arguments to be passed to the policy on creation
    :param full_tensorboard_log: (bool) enable additional logging when using tensorboard
        Note: this has no effect on TD3 logging for now
    """

    def __init__(self, policy, env, gamma=0.99, learning_rate=3e-4, buffer_size=50000,
                 learning_starts=100, train_freq=100, train_discriminator_freq=200, gradient_steps=100, batch_size=128,
                 tau=0.005, policy_delay=2, action_noise=None,
                 kernel_dim=10, noise_std=0.1, log_mmd_min=-5, log_mmd_max=0,
                 target_policy_noise=0.2, target_noise_clip=0.5,
                 random_exploration=0.0, verbose=0, tensorboard_log=None,
                 _init_setup_model=True, policy_kwargs=None, full_tensorboard_log=False,
                 using_gail=False, lfd=False, adaptive=False, explore_mode=None,
                 behavior_clone=False, using_return=False,
                 expert_data_path=None, n_jobs=1,
                 max_episode_n=4):

        super(BCQ, self).__init__(policy=policy, env=env, replay_buffer=None, verbose=verbose,
                                  policy_base=TD3Policy, requires_vec_env=False, policy_kwargs=policy_kwargs)

        self.buffer_size = buffer_size
        self.learning_rate = learning_rate
        self.learning_starts = learning_starts
        self.train_freq = train_freq
        self.batch_size = batch_size
        self.tau = tau
        self.gradient_steps = gradient_steps
        self.gamma = gamma
        self.action_noise = action_noise
        self.random_exploration = random_exploration
        self.policy_delay = policy_delay
        self.target_noise_clip = target_noise_clip
        self.target_policy_noise = target_policy_noise

        self.graph = None
        self.replay_buffer = None
        self.episode_reward = None
        self.episode_true_reward = None
        self.episode_logit = None
        self.sess = None
        self.tensorboard_log = tensorboard_log
        self.verbose = verbose
        self.params = None
        self.summary = None
        self.policy_tf = None
        self.full_tensorboard_log = full_tensorboard_log

        self.obs_target = None
        self.target_policy_tf = None
        self.actions_ph = None
        self.rewards_ph = None
        self.terminals_ph = None
        self.observations_ph = None
        self.action_target = None
        self.next_observations_ph = None
        self.step_ops = None
        self.target_ops = None
        self.infos_names = None
        self.target_params = None
        self.learning_rate_ph = None
        self.processed_obs_ph = None
        self.processed_next_obs_ph = None
        self.policy_out = None
        self.policy_train_op = None
        self.policy_loss = None

        ########## customize parallel ###################
        self.n_jobs = n_jobs

        ########## GAIL & Imitation Learning ############
        self.using_gail = using_gail
        self.lfd = lfd
        self.adaptive = adaptive
        self.explore_mode = explore_mode 
        self.discriminator = None
        self.expert_data_path = expert_data_path
        self.train_discriminator_freq = train_freq // 2
        self.hidden_size_adversary = 100
        self.vae_latent_dim = None
        self.adversary_entcoeff = 0#1e-4
        self.gradient_penalty_entcoeff = 10     
        self.d_learning_rate = 3e-4 
        self.bc_learning_rate = 3e-4 
        self.e_learning_rate = 3e-4
        self.expert_bonus = 1e-3
        self.max_episode_n = max_episode_n
        self.curiosity_start_step = -1
        self.expert_scores = []
        self.episode_buffer = TrajectoryBuffer(1e5,gamma=gamma) 
 
        self.behavior_clone=True # FOR BCQ, VAE is required
        self.bc_gradient_step=10
        self.using_return=using_return

        self.kernel_dim=kernel_dim
        self.noise_std=noise_std
        self.mmd_loss_threshold = 0.05
        self.log_mmd_max=log_mmd_max
        self.log_mmd_min=log_mmd_min

        self.d_step = 1
        self.d_batch_size = self.batch_size
        if self.using_gail:
            self.expert_dataset = ExpertDataset(expert_path=self.expert_data_path, ob_flatten=False)
            print('-'*20 + "expert_data_path: {}".format(self.expert_data_path))
            time.sleep(4)
            n_samples = len(self.expert_dataset.observations)
            #self.demo_buffer_size = 3e5
            self.demo_buffer_size = 1e5
        
        if _init_setup_model:
            self.setup_model()
        

    def _get_pretrain_placeholders(self):
        policy = self.policy_tf
        # Rescale
        policy_out = self.policy_out * np.abs(self.action_space.low)
        return policy.obs_ph, self.actions_ph, policy_out

    def mmd_loss(self,diffs, sigma):
        diffs = -tf.abs(diffs)
        diffs = tf.reduce_mean(
            tf.reduce_mean(
            tf.exp(tf.reduce_sum(diffs, -1) / (2.0*sigma)),
            axis=1),
            axis=1)
        return diffs


    def mmd_loss_laplacian(self, samples1, samples2, sigma=10):
        """MMD Loss with Laplacian kernel for matching supports""" # B * N * d
        diff_x_x = tf.expand_dims(samples1, 2) - tf.expand_dims(samples1, 1)  # B * N * N * d
        diff_x_x = self.mmd_loss(diff_x_x,sigma) # B * 1 


        diff_x_y = tf.expand_dims(samples1, 2) - tf.expand_dims(samples2, 1)
        diff_x_y = self.mmd_loss(diff_x_y,sigma)

        diff_y_y = tf.expand_dims(samples2, 2) - tf.expand_dims(samples2, 1)
        diff_y_y = self.mmd_loss(diff_y_y,sigma)


        overall_loss = tf.sqrt(diff_x_x + diff_y_y - 2.0 * diff_x_y + 1e-6)
        return overall_loss


    def setup_model(self):
        with SetVerbosity(self.verbose):
            self.graph = tf.Graph()
            with self.graph.as_default():
                n_cpu = multiprocessing.cpu_count()
                if sys.platform == 'darwin':
                    n_cpu //= 2
                cpu_slice_num = max(1, n_cpu // self.n_jobs)
                print("Total CPU:{}, CPU each job: {}".format(n_cpu, cpu_slice_num))
                self.sess = tf_util.make_session(num_cpu=cpu_slice_num, graph=self.graph)
                # initialize Curiosity classifier
                
                # initialize GAIL discriminator
                if self.using_gail:
                    self.discriminator = DiscriminatorCalssifier( 
                        self.observation_space,
                        self.action_space,
                        self.hidden_size_adversary,
                        entcoeff=self.adversary_entcoeff,
                        gradcoeff=self.gradient_penalty_entcoeff)
                
                self.bc = VAE(
                    self.observation_space,
                    self.action_space,
                    self.vae_latent_dim,
                    self.hidden_size_adversary
                ) 

                if self.using_gail:
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
                    self.target_policy_tf = self.policy(self.sess, self.observation_space, self.action_space,
                                                        **self.policy_kwargs)

                    # Initialize Placeholders
                    self.observations_ph = self.policy_tf.obs_ph
                    # Normalized observation for pixels
                    self.processed_obs_ph = self.policy_tf.processed_obs
                    self.next_observations_ph = self.target_policy_tf.obs_ph
                    self.processed_next_obs_ph = self.target_policy_tf.processed_obs
                    self.action_target = self.target_policy_tf.action_ph
                    self.terminals_ph = tf.placeholder(tf.float32, shape=(None, 1), name='terminals')
                    self.rewards_ph = tf.placeholder(tf.float32, shape=(None, 1), name='rewards')
                    self.actions_ph = tf.placeholder(tf.float32, shape=(None,) + self.action_space.shape,
                                                     name='actions')
                    self.learning_rate_ph = tf.placeholder(tf.float32, [], name="learning_rate_ph")
                    self.d_learning_rate_ph = tf.placeholder(tf.float32, [], name="d_learning_rate_ph")
                    self.bc_learning_rate_ph = tf.placeholder(tf.float32, [], name="bc_learning_rate_ph")
                    if self.behavior_clone:
                        self.bc_ob_ph = self.bc.obs_ph
                        self.bc_next_obs_ph = self.bc.next_obs_ph
                        self.bc_decoded_output = self.bc.decoded_output
                        self.bc_decoded_next_output = self.bc.decoded_next_output
                         
                if self.behavior_clone:
                    with tf.variable_scope("lagrange", reuse=False):
                        bc_log_lagrange = tf.Variable(self.log_mmd_min, trainable=True)

                    with tf.variable_scope("bc_loss", reuse=False):
                        bc_optimizer = tf.train.AdamOptimizer(learning_rate=self.bc_learning_rate_ph) 
                        self.bc_total_loss = self.bc.total_loss 
                        self.bc_losses = self.bc.losses
                        self.bc_train_op = bc_optimizer.minimize(self.bc_total_loss, var_list=self.bc.get_trainable_variables())
                        # given observation, generate policy output from a VAE decoder
                        self.bc_predicts = self.bc.multiple_decoded_output

                        bc_scalar_summaries = []
                        for i, loss_name in enumerate(self.bc.loss_names):
                            l = tf.summary.scalar(loss_name, self.bc_losses[i])
                            bc_scalar_summaries.append(l)
                        self.bc_summary = tf.summary.merge(bc_scalar_summaries)



                with tf.variable_scope("model", reuse=False):
                    # Create the policy
                    self.policy_out = policy_out = self.policy_tf.make_bcq_actor(self.processed_obs_ph, self.bc_decoded_output)
                    # Use two Q-functions to improve performance by reducing overestimation bias
                    qf1, qf2 = self.policy_tf.make_critics(self.processed_obs_ph, self.actions_ph)
                    # Q value when following the current policy
                    qf1_pi, _ = self.policy_tf.make_critics(self.processed_obs_ph,
                                                            policy_out, reuse=True)

                with tf.variable_scope("target", reuse=False):
                    # Create target networks
                    target_policy_out = self.target_policy_tf.make_bcq_actor(self.processed_next_obs_ph, self.bc_decoded_next_output)
                    # Target policy smoothing, by adding clipped noise to target actions
                    target_noise = tf.random_normal(tf.shape(target_policy_out), stddev=self.target_policy_noise)
                    target_noise = tf.clip_by_value(target_noise, -self.target_noise_clip, self.target_noise_clip)
                    # Clip the noisy action to remain in the bounds [-1, 1] (output of a tanh)
                    noisy_target_action = tf.clip_by_value(target_policy_out + target_noise, -1, 1)
                    # Q values when following the target policy
                    qf1_target, qf2_target = self.target_policy_tf.make_critics(self.processed_next_obs_ph,
                                                                                noisy_target_action)

                if self.using_gail:
                    with tf.variable_scope("discriminator_loss", reuse=False):
                        gail_optimizer = tf.train.AdamOptimizer(learning_rate=self.d_learning_rate_ph)
                        self.gail_item_loss = self.discriminator.losses
                        self.gail_total_loss = self.discriminator.total_loss
                        self.gail_sample_loss = self.discriminator.sample_loss
                        self.gail_train_op = gail_optimizer.minimize(self.gail_total_loss, var_list=self.discriminator.get_trainable_variables())
                        # Log discriminator scalars for debugging purposes
                        gail_scalar_summaries = []
                        for i, loss_name in enumerate(self.discriminator.loss_name):
                            i = tf.summary.scalar(loss_name, self.gail_item_loss[i])
                            gail_scalar_summaries.append(i)
                        self.gail_summary = tf.summary.merge(gail_scalar_summaries)
                        
                with tf.variable_scope("loss", reuse=False):
                    # Take the min of the two target Q-Values (clipped Double-Q Learning)
                    min_qf_target = tf.minimum(qf1_target, qf2_target)

                    # Targets for Q value regression
                    q_backup = tf.stop_gradient(
                        self.rewards_ph +
                        (1 - self.terminals_ph) * self.gamma * min_qf_target
                    )

                    # Compute Q-Function loss
                    td_qf1_loss = (q_backup - qf1) ** 2 
                    td_qf2_loss = (q_backup - qf2)** 2

                    qf1_loss = tf.reduce_mean(td_qf1_loss)
                    qf2_loss = tf.reduce_mean(td_qf2_loss)

                    qvalues_losses = qf1_loss + qf2_loss

                    if self.behavior_clone:
                        # Policy loss: maximise q value & minimize MMD loss -- Judy
                        # calculate mmd loss 
                        behavior_output = self.predict_multiple(self.kernel_dim) # output from behavior policy
                        vae_output = self.bc_predicts   # output from a vae decoder (imiating expert)
                        mmd_losses = self.mmd_loss_laplacian(behavior_output, vae_output, sigma=self.kernel_dim)

                        self.bc_log_lagrange = tf.clip_by_value(bc_log_lagrange, self.log_mmd_min, self.log_mmd_max)
                        policy_loss = tf.reduce_mean( -qf1_pi + tf.exp(self.bc_log_lagrange) * mmd_losses)
                        self.mmd_losses = tf.reduce_mean(mmd_losses)
                        self.policy_loss = -tf.reduce_mean(qf1_pi) 
                        
                        # train lagrange_lambda -- Judy
                        lagrange_loss = tf.reduce_mean( -qf1_pi + tf.exp(self.bc_log_lagrange) * (mmd_losses - self.mmd_loss_threshold))
                        lagrange_loss = -lagrange_loss
                        lagrange_optimizer = tf.train.AdamOptimizer(learning_rate=3e-4) 
                        self.lagrange_train_op = lagrange_optimizer.minimize(lagrange_loss, var_list=get_vars('lagrange'))
                    else:
                        # Policy loss: maximise q value
                        self.policy_loss = policy_loss = -tf.reduce_mean(qf1_pi)

                    # Policy train op
                    # will be called only every n training steps,
                    # where n is the policy delay
                    policy_optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate_ph)
                    policy_train_op = policy_optimizer.minimize(policy_loss, var_list=get_vars('model/pi'))
                    self.policy_train_op = policy_train_op

                    # Q Values optimizer
                    qvalues_optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate_ph)
                    qvalues_params = get_vars('model/values_fn/')

                    # Q Values and policy target params
                    source_params = get_vars("model/")
                    target_params = get_vars("target/")

                    # Polyak averaging for target variables
                    self.target_ops = [
                        tf.assign(target, (1 - self.tau) * target + self.tau * source)
                        for target, source in zip(target_params, source_params)
                    ]

                    # Initializing target to match source variables
                    target_init_op = [
                        tf.assign(target, source)
                        for target, source in zip(target_params, source_params)
                    ]

                    train_values_op = qvalues_optimizer.minimize(qvalues_losses, var_list=qvalues_params)

                    self.infos_names = ['qf1_loss', 'qf2_loss']
                    # All ops to call during one training step
                    self.step_ops = [qf1_loss, qf2_loss,
                                     qf1, qf2, train_values_op]

                    # Monitor losses and entropy in tensorboard
                    train_scalar_summaries = []
                    i = tf.summary.scalar('batch_mean_rewards', tf.reduce_mean(self.rewards_ph))
                    train_scalar_summaries.append(i)
                    i = tf.summary.scalar('policy_loss', policy_loss)
                    train_scalar_summaries.append(i)
                    i = tf.summary.scalar('qf1_loss', qf1_loss)
                    train_scalar_summaries.append(i)
                    i = tf.summary.scalar('qf2_loss', qf2_loss)
                    train_scalar_summaries.append(i)
                    i = tf.summary.scalar('learning_rate', tf.reduce_mean(self.learning_rate_ph))
                    train_scalar_summaries.append(i)
                    if self.behavior_clone:
                        i = tf.summary.scalar('pure_policy_loss', self.policy_loss)
                        train_scalar_summaries.append(i)
                        i = tf.summary.scalar('mmd_loss', self.mmd_losses)
                        train_scalar_summaries.append(i)
                        i = tf.summary.scalar('bc_log_lagrange', self.bc_log_lagrange)
                        train_scalar_summaries.append(i)


                # Retrieve parameters that must be saved
                self.params = get_vars("model")
                self.target_params = get_vars("target/")

                # Initialize Variables and target network
                with self.sess.as_default():
                    self.sess.run(tf.global_variables_initializer())
                    self.sess.run(target_init_op)

                #self.summary = tf.summary.merge_all()
                self.summary = tf.summary.merge(train_scalar_summaries)

    def _train_bc(self, writer, logger, current_lr, step, train_steps):

        logger.log("Optimizing Behavior Clonning classifier...")
        #logger.log(fmt_row(13, self.bc.loss_names ))
        bc_losses = []  # list of tuples, each of which gives the loss for a minibatch
        for i in range(int(train_steps)):
            batch = self.generate_train_data()
            ob_batch, ac_batch = batch[:2]

            if self.bc.normalize: 
                self.bc.obs_rms.update(ob_batch,sess=self.sess) 

            feed_dict = {
                self.bc.obs_ph: ob_batch, 
                self.bc.acs_ph: ac_batch, 
                self.bc_learning_rate_ph: current_lr
            }

            if writer is not None:
                run_ops = [self.bc_summary] + [self.bc_train_op] + self.bc_losses
                out = self.sess.run(run_ops, feed_dict)
                summary = out.pop(0)
                writer.add_summary(summary, step)
            else:
                run_ops = [self.bc_train_op] + self.bc_losses 
                out = self.sess.run(run_ops, feed_dict)  
            #### 昏哥线 ###  
            loss_dim = len(self.bc_losses)
            losses = out[-loss_dim:]
            bc_losses.append(losses)
            del batch 
            if i % 100 == 0:
                logger.log(fmt_row(13, np.mean(bc_losses, axis=0)))

    
    def generate_train_data(self, mode=None):
        # Sample a batch from the replay buffer
        expert_batch_size = self_batch_size = self.batch_size // 2
        self_batch_size = int(self.batch_size * 0.9)
        expert_batch_size = int(self.batch_size * 0.1)

        if self.lfd: # sample both self-data and expert data
            expert_batch = self.demo_replay_buffer.sample(expert_batch_size)
            self_batch = self.replay_buffer.sample(self_batch_size)
            batch = [np.concatenate((expert_batch[i],self_batch[i]),axis=0) for i in range(len(self_batch))] # numpy array
        else:
            batch = self.replay_buffer.sample(self.batch_size)
        # # Reshape actions if needed when using discrete actions
        # if isinstance(self.action_space, gym.spaces.Discrete):
        #     if len(ac_batch.shape) == 2:
        #         ac_batch = ac_batch[:, 0]
        return batch

    def _train_step(self, step, writer, learning_rate, update_policy):
        batch = self.generate_train_data()
        batch_obs, batch_actions, batch_rewards, batch_next_obs, batch_dones, batch_if_demos, batch_returns, batch_true_rewards = batch[:8]
        #############################################
        # use rewards from GAIL
        ##############################################
        if self.using_gail:
            rewards = self.get_imitate_reward(batch_obs, batch_actions, batch_rewards, batch_if_demos)
            if self.using_return:
                rewards = self.combine_both_rewards(rewards, batch_true_rewards, step, coeff=1)
        else:
            rewards = batch_rewards 

        batch_rewards = np.array(rewards).reshape(-1,1) 

        feed_dict = {
            self.observations_ph: batch_obs,
            self.actions_ph: batch_actions,
            self.next_observations_ph: batch_next_obs,
            self.rewards_ph: batch_rewards,
            self.terminals_ph: batch_dones,
            self.learning_rate_ph: learning_rate
        }
        if self.behavior_clone: 
            feed_dict[self.bc_ob_ph] = batch_obs
            feed_dict[self.bc_next_obs_ph] = batch_next_obs

        step_ops = self.step_ops 
        if update_policy:
            # Update policy and target networks
            step_ops = step_ops + [self.policy_train_op, self.target_ops, self.policy_loss]
            # Update lagrange multiplier
            if self.behavior_clone:
                step_ops = step_ops + [self.lagrange_train_op]

        # Do one gradient step
        # and optionally compute log for tensorboard
        if writer is not None:
            out = self.sess.run([self.summary] + step_ops, feed_dict)
            summary = out.pop(0)
            writer.add_summary(summary, step)
        else:
            out = self.sess.run(step_ops, feed_dict)

        # Unpack to monitor losses
        qf1_loss, qf2_loss, *_values = out
        
        del batch
        return qf1_loss, qf2_loss

    def shift_actions(self, ac_expert):
        # add noise to expert actions if needed when explore_expert is True
        if self.explore_mode is not None and self.action_noise is not None:
            if self.explore_mode == 'uniform':
                noises = np.array([self.action_noise() for _ in range(ac_expert.shape[0])]).reshape(ac_expert.shape)
                ac_expert = np.clip(ac_expert + noises, -1, 1)
        return ac_expert

    def generate_discriminator_data(self, mode = None):
        expert_batch = self.demo_replay_buffer.sample(self.d_batch_size )
        ob_expert, ac_expert = expert_batch[:2]
        # disturb expert actions if needed
        ac_expert = self.shift_actions(ac_expert)

        batch = self.replay_buffer.sample(self.d_batch_size )
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
        
        logger.log("Optimizing Discriminator...")
        #logger.log(fmt_row(13, self.discriminator.loss_name + ['discriminator-total-loss'] ))
        #timesteps_per_batch = len(observation)  

        # NOTE: uses only the last g step for observation
        d_losses = []  # list of tuples, each of which gives the loss for a minibatch
        for i in range(10):
            ob_expert, ac_expert, ob_batch, ac_batch = self.generate_discriminator_data(shaping_mode)

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

    def normalize_score(self, episode_score):
        min_score = self.config['max_score']
        assert min_score != 0
        return 1 + (episode_score - min_score) / np.abs(min_score)

    def initialize_expert_buffer(self):
        if self.using_gail:
            demo_obs, demo_actions, demo_rewards, demo_dones, demo_episode_scores = self.expert_dataset.get_transitions()   
            demo_starts = demo_dones
            demo_dones = np.concatenate((demo_dones[1:], np.array([1])))
            episode_lengths = np.where(demo_dones == 1)[0]
            n_samples = len(demo_obs)
            # get episode_score for each demo sample, either 0 or episode-reward
            episode_idx, n_episodes = 0, len(demo_episode_scores)
            for idx in range(n_samples-1):
                episode_score = demo_episode_scores[episode_idx] 
                episode_length = episode_lengths[episode_idx] 
                es = self.normalize_score(episode_score)

                if demo_dones[idx+1] == 1:
                    print(idx, 'episode_score for demonstration tarjectory: {}'.format(episode_score)) 
                    episode_idx += 1 
                    self.expert_scores.append(episode_score)
                    assert episode_length - idx >= 0
                
                discount_return = es * (self.gamma ** (episode_length - idx)) 

                self.demo_replay_buffer.add(demo_obs[idx], demo_actions[idx], demo_rewards[idx], demo_obs[idx + 1], float(demo_dones[idx]), 1.0, discount_return, demo_rewards[idx])
                if idx % 1000 == 0:
                    print("Adding demonstration to the replay buffer, processing {} %  ..".format(float(idx+1) * 100 / n_samples))
            self.demo_replay_buffer.add(demo_obs[-1], demo_actions[-1], demo_rewards[-1], demo_obs[-1], float(demo_dones[-1]), 1.0, es, demo_rewards[-1])

    def learn(self, total_timesteps, callback=None, seed=None,
              log_interval=4, tb_log_name="TD3", reset_num_timesteps=True, replay_wrapper=None, config=None):

        new_tb_log = self._init_num_timesteps(reset_num_timesteps)
        self.config = config
        shaping_mode = self.config['shaping_mode']
        # first, add expert-demonstrations in the replay buffer
        if self.using_gail:
            self.initialize_expert_buffer()

        if replay_wrapper is not None:
            self.demo_replay_buffer = replay_wrapper(self.demo_replay_buffer)
            self.replay_buffer = replay_wrapper(self.replay_buffer)

        with SetVerbosity(self.verbose), TensorboardWriter(self.graph, self.tensorboard_log, tb_log_name, new_tb_log) \
                as writer:

            self._setup_learn(seed)

            # Transform to callable if needed
            self.learning_rate = get_schedule_fn(self.learning_rate)
            self.d_learning_rate = get_schedule_fn(self.d_learning_rate)
            self.bc_learning_rate = get_schedule_fn(self.bc_learning_rate)
            self.e_learning_rate = get_schedule_fn(self.e_learning_rate)
            # Initial learning rate
            current_lr = self.learning_rate(1)
            current_d_lr = self.d_learning_rate(1)
            # initialize coefficient for combining reward
            self.return_schedule=LinearSchedule(
                total_timesteps,
                initial_p=0,
                final_p=1)

            start_time = time.time()
            episode_rewards = [0.0]
            episode_successes = []
            if self.action_noise is not None:
                self.action_noise.reset()
            obs = self.env.reset()
            self.episode_reward = np.zeros((1,))
            if self.using_gail:
                self.episode_true_reward = np.zeros((1,))
                self.episode_logit= np.zeros((1,))
            ep_info_buf = deque(maxlen=100)
            n_updates = 0
            infos_values = []
            episode_starts = 0 

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
                    #action = self.policy_tf.step(obs[None]).flatten()
                    vae_output = self.bc.get_decoded(obs[None], sess=self.sess)
                    feed_dict = {
                        self.bc_ob_ph: obs[None],
                    }
                    action = self.policy_tf.bcq_step(obs[None], feed_dict).flatten()

                    # Add noise to the action, as the policy
                    # is deterministic, this is required for exploration
                    if self.action_noise is not None:
                        noise = self.action_noise()
                        action = np.clip(action + noise, -1, 1)
                    # Rescale from [-1, 1] to the correct bounds
                    rescaled_action = action * np.abs(self.action_space.low)

                assert action.shape == self.env.action_space.shape
                #time.sleep(.002)
                new_obs, true_reward, done, info = self.env.step(rescaled_action) 
                reward = true_reward
                # Retrieve reward and episode length if using Monitor wrapper
                maybe_ep_info = info.get('episode')
                if maybe_ep_info is not None:
                    ep_info_buf.extend([maybe_ep_info])
                    episode_score = maybe_ep_info['r'] 
                    episode_starts = step + 1
                else:
                    episode_score = None 
                ##############################################
                # add transition to buffer
                ##############################################
                current_steps = step - episode_starts + 1
                self.episode_buffer.add(obs, action, reward, new_obs, float(done), None, episode_score, true_reward)
                

                if episode_score is not None:
                    # when to start use curiosity reward
                    if episode_score > self.config['max_score'] and self.curiosity_start_step == -1:
                        self.curiosity_start_step = step
                    # add to demo-buffer when needed
                    es = self.normalize_score(episode_score)

                    if episode_score > self.expert_scores[0]:
                        self.expert_scores.pop(0)
                        self.expert_scores.append(episode_score)
                        self.expert_scores.sort()
                        ## adjust coefficient for sythentic rewards: r^ = alpha* r_env + (1 - alpha) * r_imitation
                            

                    if self.adaptive:
                        if episode_score > self.expert_scores[0]:
                            print("Adding new trajectory with score {} to replay buffer, expert-score {} ".format(episode_score,self.expert_scores[0]))
                            for transition in self.episode_buffer.get_episode_return(es):
                                s, a, r, s1, if_done, _, _, discount_return, true_r = transition 
                                self.demo_replay_buffer.add(s, a, r, s1, if_done, 1, discount_return)
                        else:
                            for transition in self.episode_buffer.get_episode_return(es):
                                s, a, r, s1, if_done, _, _, discount_return, true_r = transition
                                self.replay_buffer.add(s, a, r, s1, if_done, 0, discount_return, true_r)
                    else:
                        # we add self-generated data to replay-buffer when (1) the score not reach threshold; or (2) not adaptive-imitating
                        for transition in self.episode_buffer.get_episode_return(es):
                            s, a, r, s1, if_done, _, _, discount_return, true_r = transition
                            self.replay_buffer.add(s, a, r, s1, if_done, 0, discount_return, true_r)
                        # reset episode buffer
                    self.episode_buffer.reset()

                obs = new_obs

                if writer is not None:
                    # Write reward per episode to tensorboard
                    ep_reward = np.array([reward]).reshape((1, -1))
                    ep_true_reward = np.array([true_reward]).reshape((1, -1))
                    ep_done = np.array([done]).reshape((1, -1))
                    #ep_logits = np.array([bonus]).reshape((1,-1))
                    self.episode_reward = total_episode_reward_logger(
                        self.episode_reward,
                        ep_reward,
                        ep_done,
                        writer,
                        self.num_timesteps)

                if step % self.train_freq == 0:
                    mb_infos_vals = []
                    frac = 1.0 - step / total_timesteps
                    current_lr = self.learning_rate(frac)
                    # Update policy, critics and target networks
                    for grad_step in range(self.gradient_steps):
                        # Break if the warmup phase is not over
                        # or if there are not enough samples in the replay buffer
                        if not (self.replay_buffer.can_sample(self.batch_size) and self.num_timesteps >= self.learning_starts):
                            break
                        n_updates += 1
                        # Compute current learning_rate
                        # Update policy and critics (q functions)
                        # Note: the policy is updated less frequently than the Q functions
                        # this is controlled by the `policy_delay` parameter
                        mb_infos_vals.append(
                            self._train_step(step, writer, current_lr, (step + grad_step) % self.policy_delay == 0))
                    
                    ##### train behavior cloning classifier
                    if self.behavior_clone:
                        current_bc_lr = self.bc_learning_rate(frac)
                        if self.replay_buffer.can_sample(self.batch_size) and self.num_timesteps >= self.learning_starts:
                            self._train_bc(writer, logger, current_bc_lr, step, self.bc_gradient_step)

                    # Log losses and entropy, useful for monitor training
                    if len(mb_infos_vals) > 0:
                        infos_values = np.mean(mb_infos_vals, axis=0)
                
                if self.using_gail and step % self.train_discriminator_freq == 0: 
                    frac = 1.0 - step / total_timesteps
                    current_d_lr = self.d_learning_rate(frac)
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
        _ = np.array(observation)

        if actions is not None:
            raise ValueError("Error: TD3 does not have action probabilities.")

        # here there are no action probabilities, as DDPG does not use a probability distribution
        warnings.warn("Warning: action probability is meaningless for TD3. Returning None")
        return None

    def predict_multiple(self, num_sample):
        actions = self.policy_out
        actions = tf.reshape(actions, (-1, self.action_space.shape[0]))
        # 1 * B * d -- > N * B * d --> B * N * d
        dup_actions = tf.expand_dims(actions, axis=0)  # 1 * B * d
        dup_actions = tf.tile(dup_actions, tf.constant([num_sample, 1, 1])) # N * B * d 
        dup_actions = tf.transpose(dup_actions,perm=[1,0,2])  # B * N * d

        noise_shape = tf.shape(dup_actions) # d * N * B
        noise = tf.random.normal(noise_shape, mean=0, stddev=self.noise_std)  
        
        actions = tf.clip_by_value(dup_actions + noise, -1, 1) 
        actions = actions * tf.abs(self.action_space.low)  # scale the output for the prediction
        return actions 

        
    def predict(self, observation, state=None, mask=None, deterministic=True):
        #print(observation.shape)
        #if len(observation.shape 
        #observation = np.array(observation)
        return None
        #vectorized_env = self._is_vectorized_observation(observation, self.observation_space)

        #observation = observation.reshape((-1,) + self.observation_space.shape)
        #actions = self.policy_tf.step(observation)

        #if self.action_noise is not None and not deterministic:
        #    actions = np.clip(actions + self.action_noise(), -1, 1)

        #actions = actions.reshape((-1,) + self.action_space.shape)  # reshape to the correct action shape
        #actions = actions * np.abs(self.action_space.low)  # scale the output for the prediction

        #if not vectorized_env:
        #    actions = actions[0]

        #return actions, None

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
            # Should we also store the replay buffer?
            # this may lead to high memory usage
            # with all transition inside
            # "replay_buffer": self.replay_buffer
            "policy_delay": self.policy_delay,
            "target_noise_clip": self.target_noise_clip,
            "target_policy_noise": self.target_policy_noise,
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


    def get_curiosity_reward(self, obs, action, next_obs, step):
        beta = self.curiosity_schedule.value(step - self.curiosity_start_step)
        bonus = beta * self.explorer.get_bonus(obs, action, next_obs, sess=self.sess)
        return bonus

    def get_imitate_reward(self, observation, action, true_reward, if_demo):
        _, p = self.discriminator.get_confidence(observation, action, sess=self.sess)
        return - np.log(1 - p + 1e-8)
    

    def combine_both_rewards(self, rewards1, rewards2, step, coeff=1):
        alpha = self.return_schedule.value(step)
        return alpha * rewards1 +  (1 - alpha) * coeff * rewards2
        
 
