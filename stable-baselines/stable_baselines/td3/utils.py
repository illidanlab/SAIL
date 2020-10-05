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
from stable_baselines.common.schedules import LinearSchedule, BlockwiseSchedule
from stable_baselines.common.vec_env import VecEnv
from stable_baselines.deepq.replay_buffer import TrajectoryBuffer, ReplayBuffer, ReplayBufferExtend, ExpertBuffer, PrioritizedReplayBufferExtend
from stable_baselines.ppo2.ppo2 import safe_mean, get_schedule_fn
from stable_baselines.sac.sac import get_vars
from stable_baselines.td3.policies import TD3Policy
from stable_baselines import logger
from stable_baselines.gail.adversary import DiscriminatorCalssifier 
from stable_baselines.gail.vae import VAE
from stable_baselines.gail.dataset.dataset import ExpertDataset
from tensorflow.contrib.layers import l2_regularizer
 
import tensorflow_probability as tfp 

ds = tfp.distributions

LOG_STD_MIN = -5
LOG_STD_MAX = 2
EPS = np.finfo(np.float32).eps

def weighted_softmax(x, weights, axis=0):
    x = x - tf.reduce_max(x, axis=axis)
    return weights * tf.exp(x) / tf.reduce_sum(weights*tf.exp(x), axis=axis, keepdims=True)


class Critics(object):
    def __init__(self, observation_space, action_space, hidden_size=256, replay_regularization=0.1,
                 gradient_penalty_entcoeff=10, gamma=0.99, scope="critics", normalize=True):
        """
        actor for ValueDice

        :param observation_space: (gym.spaces)
        :param action_space: (gym.spaces)
        :param hidden_size: ([int]) the hidden dimension for the MLP
        :param entcoeff: (float) the entropy loss weight
        :param scope: (str) tensorflow variable scope
        :param normalize: (bool) Whether to normalize the reward or not
        """
        # TODO: support images properly (using a CNN)
        self.scope = scope
        self.observation_shape = observation_space.shape
        self.actions_shape = action_space.shape
        self.l2_scale = 1e-4
        self.gamma = gamma
        self.replay_regularization=replay_regularization
        self.gradient_penalty_entcoeff=gradient_penalty_entcoeff
        #print(observation_space.dtype)

        if isinstance(action_space, gym.spaces.Box):
            # Continuous action space
            self.discrete_actions = False
            self.n_actions = action_space.shape[0]
        elif isinstance(action_space, gym.spaces.Discrete):
            self.n_actions = action_space.n
            self.discrete_actions = True
        else:
            raise ValueError('Action space not supported: {}'.format(action_space))

        self.hidden_size = hidden_size
        self.normalize = normalize
        self.obs_rms = None

        # Placeholders for training
        self.obs_ph = tf.placeholder(tf.float32, (None,) + self.observation_shape,
                                               name="observations_ph")
        self.acs_ph = tf.placeholder(action_space.dtype, (None,) + self.actions_shape,
                                               name="actions_ph")
        self.next_obs_ph = tf.placeholder(tf.float32, (None,) + self.observation_shape,
                                               name="next_obs_ph")
        #tf.placeholder(action_space.dtype, (None,) + self.actions_shape,
                                               #name="rb_next_actions_ph")

        self.expert_obs_ph = tf.placeholder(tf.float32, (None,) + self.observation_shape,
                                               name="expert_observations_ph")
        self.expert_acs_ph = tf.placeholder(action_space.dtype, (None,) + self.actions_shape,
                                               name="expert_actions_ph")
        self.expert_next_obs_ph = tf.placeholder(tf.float32, (None,) + self.observation_shape,
                                               name="expert_next_observations_ph")
        self.next_obs_fl = self.preprocess_states(self.next_obs_ph) 
        self.expert_next_obs_fl = self.preprocess_states(self.expert_next_obs_ph, reuse=True) 
        #tf.placeholder(action_space.dtype, (None,) + self.actions_shape,
         #                                      name="expert_next_actions_ph")
        #tf.placeholder(action_space.dtype, (None,) + self.actions_shape,
                                 #              name="policy_initial_actions_ph")
        #self.learning_rate_ph = tf.placeholder(tf.float32, [], name="learning_rate_ph")

    def setup_model(self, rb_next_actions, expert_next_actions):
        self.policy_initial_actions_ph = expert_next_actions 
        self.expert_next_actions_ph = expert_next_actions
        self.rb_next_actions_ph = rb_next_actions
        # Preprocessing Input
        obs_fl = self.preprocess_states(self.obs_ph, reuse=True)
        act_fl = self.preprocess_actions(self.acs_ph, reuse=False)
        next_act_fl = self.preprocess_actions(self.rb_next_actions_ph, reuse=False)

        expert_obs_fl = self.preprocess_states(self.expert_obs_ph, reuse=True)
        expert_act_fl = self.preprocess_actions(self.expert_acs_ph, reuse=True)
        expert_next_actions_fl = self.preprocess_actions(self.expert_next_actions_ph, reuse=True) 
        expert_initial_states_fl = self.preprocess_states(self.expert_obs_ph, reuse=True) 
        policy_initial_actions_fl = self.preprocess_actions(self.policy_initial_actions_ph, reuse=True)  

        with tf.GradientTape(
            watch_accessed_variables=False, persistent=True) as tape:
            # Inputs for the linear part of DualDICE loss.   
            expert_nu_0 = self.make_critics(tf.concat([expert_initial_states_fl, policy_initial_actions_fl], 1), reuse=False)

            expert_inputs = tf.concat([expert_obs_fl, expert_act_fl], 1)
            expert_nu = self.make_critics(expert_inputs, reuse=True)

            expert_next_inputs = tf.concat([self.expert_next_obs_fl, expert_next_actions_fl], 1)
            expert_nu_next = self.make_critics(expert_next_inputs, reuse=True)

            rb_inputs = tf.concat([obs_fl, act_fl], 1)
            rb_nu = self.make_critics(rb_inputs, reuse=True)

            rb_next_inputs = tf.concat([self.next_obs_fl, next_act_fl], 1)
            rb_nu_next = self.make_critics(rb_next_inputs,  reuse=True)

            expert_diff = expert_nu - self.gamma * expert_nu_next
            rb_diff = rb_nu - self.gamma * rb_nu_next

            linear_loss_expert = tf.reduce_mean(expert_nu_0 * (1 - self.gamma))

            linear_loss_rb = tf.reduce_mean(rb_diff)

            rb_expert_diff = tf.concat([expert_diff, rb_diff], 0)
            rb_expert_weights = tf.concat([
                tf.ones(tf.shape(expert_diff)) * (1 - self.replay_regularization),
                tf.ones(tf.shape(rb_diff)) * self.replay_regularization
            ], 0)

            rb_expert_weights /= tf.reduce_sum(rb_expert_weights)
            non_linear_loss = tf.reduce_sum(
                tf.stop_gradient(
                    weighted_softmax(rb_expert_diff, rb_expert_weights, axis=0)) *
                rb_expert_diff)

            linear_loss = (
                linear_loss_expert * (1 - self.replay_regularization) +
                linear_loss_rb * self.replay_regularization)

            loss = (non_linear_loss - linear_loss)

            alpha_shape = self.observation_shape[0] + self.actions_shape[0]
            alpha = np.random.uniform(size=(1, alpha_shape))

            nu_inter = alpha * expert_inputs + (1 - alpha) * rb_inputs
            nu_next_inter = alpha * expert_next_inputs + (1 - alpha) * rb_next_inputs

            nu_inter = tf.concat([nu_inter, nu_next_inter], 0)    

            with tf.GradientTape(watch_accessed_variables=False) as tape2:
                tape2.watch(nu_inter)
                nu_output = self.make_critics(nu_inter,reuse=True)
            nu_grad = tape2.gradient(nu_output, [nu_inter])[0] + EPS
            nu_grad_penalty = tf.reduce_mean(
                tf.square(tf.norm(nu_grad, axis=-1) - 1))
        
        nu_loss = loss + nu_grad_penalty * self.gradient_penalty_entcoeff
        expert_diff_loss = tf.reduce_mean(expert_diff)
        rb_diff_loss = tf.reduce_mean(rb_diff)
        self.variables = self.get_trainable_variables()

        # Loss terms
        self.losses = [nu_loss, loss, nu_grad_penalty, expert_diff_loss, rb_diff_loss]
        self.loss_name = ["nu_loss", "loss", "nu_grad_penalty", "expert_diff", "rb_diff"]
        self.total_loss = nu_loss
        self.dice_loss = loss 
        self.reward_op = - rb_diff 

    def get_reward(self, obs, actions, next_obs, next_actions, sess=None):
        """
        Predict the reward using the observation and action

        :param obs: (tf.Tensor or np.ndarray) the observation
        :param actions: (tf.Tensor or np.ndarray) the action
        :return: (np.ndarray) the reward
        """
        if sess is None:
            sess = tf.get_default_session()
        
        if len(obs.shape) == 1:
            obs = np.expand_dims(obs, 0)
        if len(actions.shape) == 1:
            actions = np.expand_dims(actions, 0)
        elif len(actions.shape) == 0:
            # one discrete action
            actions = np.expand_dims(actions, 0)

        feed_dict = {
            self.obs_ph: obs,
            self.acs_ph: actions,
            self.next_obs_ph: next_obs,
            self.rb_next_actions_ph:next_actions}
        reward = sess.run(self.reward_op, feed_dict)
        return reward


    
    def make_critics(self, inputs, reuse=False):
        with tf.variable_scope(self.scope):
            if reuse:
                tf.get_variable_scope().reuse_variables()
            p_h1 = tf.contrib.layers.fully_connected(inputs, self.hidden_size, activation_fn=tf.nn.relu)
            p_h2 = tf.contrib.layers.fully_connected(p_h1, self.hidden_size, activation_fn=tf.nn.relu) 
            logits = tf.contrib.layers.fully_connected(p_h2, 1, activation_fn=tf.identity) 
        return logits

    def preprocess_states(self, obs_ph, reuse=False):
        with tf.variable_scope(self.scope):
            if reuse:
                tf.get_variable_scope().reuse_variables()
            if self.normalize:
                with tf.variable_scope("critic_obfilter"):
                    self.obs_rms = RunningMeanStd(shape=self.observation_shape)
                obs = (obs_ph - self.obs_rms.mean) / self.obs_rms.std
            else:
                obs = obs_ph
            ob_fl = tf.contrib.layers.flatten(obs)
        return ob_fl

    def preprocess_actions(self, acs_ph, reuse=False):
        with tf.variable_scope(self.scope):
            if reuse:
                tf.get_variable_scope().reuse_variables()
            if self.discrete_actions:
                one_hot_actions = tf.one_hot(acs_ph, self.n_actions)
                actions_ph = tf.cast(one_hot_actions, tf.float32)
            else:
                actions_ph = acs_ph
            act_fl = tf.contrib.layers.flatten(actions_ph)
        return act_fl


    def get_trainable_variables(self):
        """
        Get all the trainable variables from the graph

        :return: ([tf.Tensor]) the variables
        """
        return tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, self.scope)