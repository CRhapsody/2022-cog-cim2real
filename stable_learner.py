import os.path
import time

from stable_baselines3 import PPO, DDPG, A2C, TD3
from nav_gym_env import NavEnvironment
from Cogenvdecoder.CogEnvDecoder import CogEnvDecoder
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3.common.noise import NormalActionNoise
from nav_model import NavActorCriticPolicy
import numpy as np


baseline_models_path = "C:\\Users\\Administrator\\PycharmProjects\\2022-cog-cim2real\\stable_baseline_models"


def init_env_vec(worker_numbers):
    env_fns = []
    for worker_number in worker_numbers:
        def new_env(inner_worker_number):
            return lambda :init_env(inner_worker_number)
        init_env_fn = new_env(worker_number)
        env_fns.append(init_env_fn)
    return SubprocVecEnv(env_fns)


def init_env(worker_number=1):
    cog_env = CogEnvDecoder(
        env_name="C:\\Users\\Administrator\\PycharmProjects\\2022-cog-cim2real\\win_v2.1\\cog_sim2real_env.exe",
        no_graphics=False, time_scale=3, worker_id=worker_number)
    env = NavEnvironment(Cog_env=cog_env, env_id=worker_number)
    return env


def init_model(env):
    # Custom actor and critic architecture
    # Create the agent
    action_noise = NormalActionNoise(mean=np.zeros(1), sigma=0.1*np.ones(1))
    model = PPO(NavActorCriticPolicy,
                env,
                verbose=1,
                tensorboard_log="./log/",
                )
    # model = TD3("MlpPolicy",
    #             env,
    #             policy_kwargs=dict(net_arch=dict(pi=[256, 256], qf=[512, 256])),
    #             verbose=1,
    #             tensorboard_log="./log/",
    #             gamma=1,
    #             action_noise=action_noise
    #             )
    # model = A2C("MlpPolicy",
    #             env,
    #             policy_kwargs=dict(net_arch=[512, 512, 256]),
    #             verbose=1,
    #             tensorboard_log="./log/",
    #             )
    return model


def train(model):
    total_steps = 100_000_000
    eval_interval = 50000
    i = 0
    while i < total_steps:
        print(f"current step: {i}")
        print(f"start training for {eval_interval} steps.")
        model.learn(eval_interval, log_interval=100)
        i += eval_interval
        print("start eval")
        timeArray = time.localtime(time.time())
        t = time.strftime("%m_%d_%H_%M_%S", timeArray)
        p = os.path.join(baseline_models_path, "model_" + t)
        model.save(p)
        evaluate(p)


def evaluate(model_path, rename_with_reward=True):
    env = init_env(worker_number=100)
    model = init_model(env)
    model.load(model_path)
    mean_reward = evaluate_model(model)
    print(f"model_path: {model_path}, mean_reward: {mean_reward}")
    if rename_with_reward:
        os.rename(model_path + ".zip", model_path + f"_mean_reward_{mean_reward}.zip")
    env.close()


def evaluate_model(model):
    mean_reward, _ = evaluate_policy(model, model.get_env(), n_eval_episodes=1)
    return mean_reward


def learn():
    # env = init_env(1)
    env = init_env_vec([1, 2, 3])
    model = init_model(env)
    train(model)


def continual_learn(model_path):
    env = init_env_vec([1])
    model = init_model(env)
    model.load(model_path)
    train(model)


def test():
    env = init_env(1)
    model = init_model(env)
    mean_reward = evaluate_model(model)
    print(f"mean_reward: {mean_reward}")


def main():
    learn()
    # continual_learn("stable_baseline_models/model_05_30_09_32_45_mean_reward_-65561.360782")


if __name__ == '__main__':
    # test()
    main()
