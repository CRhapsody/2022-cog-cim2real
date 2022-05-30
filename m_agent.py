import base64
from gc import collect
# import imageio
# import IPython
# import matplotlib.pyplot as plt
import os
# import reverb
import tempfile
import PIL.Image

import tensorflow as tf

from tf_agents.agents.ddpg import critic_network
from tf_agents.agents.sac import sac_agent
from tf_agents.agents.sac import tanh_normal_projection_network
# from tf_agents.environments import suite_pybullet
from tf_agents.metrics import py_metrics
from tf_agents.networks import actor_distribution_network
from tf_agents.policies import greedy_policy
from tf_agents.policies import py_tf_eager_policy
from tf_agents.policies import random_py_policy,random_tf_policy
from tf_agents.replay_buffers import reverb_replay_buffer
from tf_agents.replay_buffers import tf_uniform_replay_buffer
from tf_agents.environments import tf_environment

from tf_agents.replay_buffers import reverb_utils
from tf_agents.train import actor
from tf_agents.train import learner
from tf_agents.train import triggers
from tf_agents.train.utils import spec_utils
from tf_agents.train.utils import strategy_utils
from tf_agents.train.utils import train_utils
from tf_agents.environments import tf_py_environment
from tf_agents.drivers import dynamic_step_driver

from environment import Environment
from Cogenvdecoder.CogEnvDecoder import CogEnvDecoder
from parameter_list import *
use_gpu = True

tempdir = tempfile.gettempdir()


class Agent():
    def __init__(self) -> None:
        collect_Cog_env = CogEnvDecoder(env_name="C:\\Users\\Administrator\\PycharmProjects\\2022-cog-cim2real\\win_v2.1\\cog_sim2real_env.exe",
                                        no_graphics=False,
                                        time_scale=1,
                                        worker_id=1)
        self.py_collect_env = Environment(Cog_env= collect_Cog_env)
        self.tf_collect_env = tf_py_environment.TFPyEnvironment(self.py_collect_env)

        # eval_Cog_env = CogEnvDecoder(env_name="win_v1/RealGame.exe", no_graphics=False, time_scale=1, worker_id=1)
        # self.eval_env = Environment(Cog_env= eval_Cog_env)
        # self.eval_env = tf_py_environment.TFPyEnvironment(self.eval_env)

        self.strategy = strategy_utils.get_strategy(tpu=False, use_gpu=use_gpu)

        self.actor_net = self.init_actor_net()
        self.critic_net = self.init_critic_net()
        self.train_step, self.tf_agent = self.init_tf_agents()

        self.replay_buffer = self.init_replay_buffer() #self.reverb_server,
        self.experience_dataset_fn = self.get_dataset()

        self.collect_policy = self.get_collect_policy()
        self.eval_policy = self.get_eval_policy()
        self.random_policy = random_tf_policy.RandomTFPolicy(
                self.tf_collect_env.time_step_spec(), self.tf_collect_env.action_spec())

        # self.rb_observer = self.init_rb_observer()
        self.run_init_actor()
        self.collect_actor = self.init_collect_actor()
        self.eval_actor = self.init_eval_actor()

        self.agent_learner = self.init_learner()


    def init_critic_net(self):
        observation_spec, action_spec, time_step_spec = (
                                    spec_utils.get_tensor_specs(self.py_collect_env))

        with self.strategy.scope():
            critic_net = critic_network.CriticNetwork(
            (observation_spec, action_spec),
            observation_fc_layer_params=None,
            action_fc_layer_params=None,
            joint_fc_layer_params=critic_joint_fc_layer_params,
            kernel_initializer='glorot_uniform',
            last_kernel_initializer='glorot_uniform')
        
        return critic_net
    

    def init_actor_net(self):
        observation_spec, action_spec, time_step_spec = (
                                        spec_utils.get_tensor_specs(self.py_collect_env))
        with self.strategy.scope():
            actor_net = actor_distribution_network.ActorDistributionNetwork(
            observation_spec,
            action_spec,
            fc_layer_params=actor_fc_layer_params,
            continuous_projection_net=(
            tanh_normal_projection_network.TanhNormalProjectionNetwork))
        
        return actor_net
    
    def init_tf_agents(self):
        observation_spec, action_spec, time_step_spec = (
                                        spec_utils.get_tensor_specs(self.py_collect_env))

        with self.strategy.scope():
            train_step = train_utils.create_train_step()

            tf_agent = sac_agent.SacAgent(
                    time_step_spec,
                    action_spec,
                    actor_network=self.actor_net,
                    critic_network=self.critic_net,
                    actor_optimizer=tf.keras.optimizers.Adam(
                        learning_rate=actor_learning_rate),
                    critic_optimizer=tf.keras.optimizers.Adam(
                        learning_rate=critic_learning_rate),
                    alpha_optimizer=tf.keras.optimizers.Adam(
                        learning_rate=alpha_learning_rate),
                    target_update_tau=target_update_tau,
                    target_update_period=target_update_period,
                    td_errors_loss_fn=tf.math.squared_difference,
                    gamma=gamma,
                    reward_scale_factor=reward_scale_factor,
                    train_step_counter=train_step)
        
            tf_agent.initialize()
        
        return train_step, tf_agent



    #建立 reverb replay buffer
    def init_replay_buffer(self,table_name = 'uniform_table',sequence_length=2):

        replay_buffer = tf_uniform_replay_buffer.TFUniformReplayBuffer(
                self.tf_agent.collect_data_spec,
                batch_size = self.tf_collect_env.batch_size)
        
        return replay_buffer


    
    def get_dataset(self):
        '''
        通过 reverb replay buffer 里的数据生成dataset
        '''
        dataset = self.replay_buffer.as_dataset(
                                        sample_batch_size=batch_size, num_steps=2).prefetch(50)
        experience_dataset_fn = lambda: dataset
        return experience_dataset_fn

    def get_eval_policy(self):
        '''
        
        '''
        tf_eval_policy = self.tf_agent.policy
        eval_policy = py_tf_eager_policy.PyTFEagerPolicy(tf_eval_policy, use_tf_function=True)
        return eval_policy
    
    def get_collect_policy(self):
        '''
        
        '''
        tf_collect_policy = self.tf_agent.policy
        # collect_policy = py_tf_eager_policy.PyTFEagerPolicy(tf_collect_policy, use_tf_function=True)
        return tf_collect_policy


    # def init_rb_observer(self,table_name = 'uniform_table',sequence_length=2,
    #                 stride_length=1):
    #     '''
    #     sequence_length指储存轨迹的长度,stride_length指轨迹之间窗口overlap的长度
    #     '''
    #     rb_observer = reverb_utils.ReverbAddTrajectoryObserver(
    #                 self.replay_buffer.py_client,
    #                 table_name,
    #                 sequence_length=sequence_length,
    #                 stride_length=stride_length)
    #     return rb_observer
    
    def make_actor(self,env,policy):
        '''
        
        '''
        # return actor.Actor(env,
        #             policy,
        #             self.train_step,
        #             steps_per_run=initial_collect_steps,
        #             observers=[self.replay_buffer.add_batch]) #self.rb_observer
        return dynamic_step_driver.DynamicStepDriver(
                    env,
                    policy,
                    num_steps = initial_collect_steps,
                    observers = [self.replay_buffer.add_batch])

    
    def run_init_actor(self):
        '''
        运行初始actor, 使用随即策略在训练过程中收集初始数据
        '''
        initial_collect_actor = self.make_actor(self.tf_collect_env, self.random_policy)
        initial_collect_actor.run()


    
    def init_collect_actor(self):
        '''
        运行收集actor, 使用收集策略在训练过程中收集数据
        '''
        # env_step_metric = py_metrics.EnvironmentSteps()
        # collect_actor = actor.Actor(
        #                 self.py_collect_env,
        #                 self.collect_policy,
        #                 self.train_step,
        #                 steps_per_run=1,
        #                 metrics=actor.collect_metrics(10),
        #                 summary_dir=os.path.join(tempdir, learner.TRAIN_DIR),
        #                 observers=[self.replay_buffer.add_batch, env_step_metric])
        driver = dynamic_step_driver.DynamicStepDriver(
                    env = self.tf_collect_env,
                    policy = self.tf_agent.collect_policy,
                    num_steps = 500,
                    observers = [self.replay_buffer.add_batch])
        return driver
    
    
    def init_eval_actor(self):
        '''
        运行eval actor, 对训练中的策略进行估计
        '''
        eval_actor = actor.Actor(
            self.py_collect_env,
            self.eval_policy,
            self.train_step,
            steps_per_run = num_eval_episodes,
            metrics=actor.eval_metrics(num_eval_episodes),
            summary_dir=os.path.join(tempdir, 'eval'),)
        return eval_actor

    def init_learner(self):
        '''
        learner包含agent, 并且可以在训练过程中使用replay_buffer里的数据对策略参数进行梯度下降
        '''
        saved_model_dir = os.path.join(tempdir, learner.POLICY_SAVED_MODEL_DIR)
        learning_triggers = [
                triggers.PolicySavedModelTrigger(
                saved_model_dir,
                self.tf_agent,
                self.train_step,
                interval=policy_save_interval),
                triggers.StepPerSecondLogTrigger(self.train_step, interval=1000),
        ]
        agent_learner = learner.Learner(
                tempdir,
                self.train_step,
                self.tf_agent,
                self.experience_dataset_fn,
                triggers=learning_triggers,
                strategy=self.strategy)
        return agent_learner

    def get_eval_metrics(self):
        self.eval_actor.run()
        results = {}
        for metric in self.eval_actor.metrics:
            results[metric.name] = metric.result()
        return results
    
    def print_log_eval_metrics(self, step, metrics):
        eval_results = (', ').join(
            '{} = {:.6f}'.format(name, result) for name, result in metrics.items())
        print('step = {0}: {1}'.format(step, eval_results))
    

    def train(self):
        '''
        return the evaluation return as a list
        '''
    


        # Reset the train step
        self.tf_agent.train_step_counter.assign(0)

        # Evaluate the agent's policy once before training.
        avg_return = self.get_eval_metrics()["AverageReturn"]
        eval_returns = [avg_return]


        for j in range(num_iterations):
            # reset the environment
            py_time_step = self.py_collect_env.reset()

            # Training.
            self.collect_actor.run()
            loss_info = self.agent_learner.run(iterations=1)

            # Evaluating.
            step = self.agent_learner.train_step_numpy

            if eval_interval and step % eval_interval == 0:
                metrics = self.get_eval_metrics()
                self.print_log_eval_metrics(step, metrics)
                eval_returns.append(metrics["AverageReturn"])

            if log_interval and step % log_interval == 0:
                print('step = {0}: loss = {1}'.format(step, loss_info.loss.numpy()))

        # self.rb_observer.close()
        # self.reverb_server.stop()

        return eval_returns
    
    def eval(self):
        pass


    
    # def get_spec(env):
    #     observation_spec, action_spec, time_step_spec = (
    #                                     spec_utils.get_tensor_specs(env))

    #     return observation_spec, action_spec, time_step_spec


def main():
    agent = Agent()
    agent.train()

if __name__ == "__main__":
    main()


    

