

# from __future__ import absolute_import
# from __future__ import division
# from __future__ import print_function

from tkinter import _flatten
from gym import Env
from CogEnvDecoder import CogEnvDecoder
import abc
import tensorflow as tf

import numpy as np

from tf_agents.environments import py_environment
from tf_agents.environments import tf_environment
from tf_agents.environments import tf_py_environment
from tf_agents.environments import utils
from tf_agents.specs import array_spec
from tf_agents.environments import wrappers
from tf_agents.environments import suite_gym
from tf_agents.trajectories import time_step as ts
import math

class Environment(py_environment.PyEnvironment):
    def __init__(self, Cog_env : CogEnvDecoder) -> None:
        self._Cog_env = Cog_env

        self._action_spec = array_spec.BoundedArraySpec( # [vx, vy, vw, fire]
            shape = (4,), dtype = np.float32, minimum=[-2,-2,-math.pi,0], maximum=[2,2,math.pi,1], name = "action"
        )
        self._observation_spec = array_spec.BoundedArraySpec(
            shape = (89,), dtype = np.float32, name = "observation"
        )
        self._state = 0
        self_episode_ended = False

    @classmethod
    def transform_state(cls, state, has_image = False):

        if has_image:
            return {"image": state["image"], "else": state["vector"] + state["laser"]}
        else:
            # list(_flatten(state["vector"]))
            # temp_state = np.append(state["vector"],state["laser"],axis = 0)
            return np.array(list(_flatten(state["vector"]))+state["laser"].tolist(), dtype=np.float32)
    
    def action_spec(self):
        return self._action_spec 

    def observation_spec(self):
        return self._observation_spec


    def _reset(self):
        state = self._Cog_env.reset()
        
        # self._state = state["vector"] + state["laser"]
        self._state = Environment.transform_state(state)

        # self._episode_ended = False
        return ts.restart(self._state) 

    def _step(self, action):

        state, reward, done, [info, judge_result] = self._Cog_env.step(action)
        self._state = Environment.transform_state(state)
        


        return ts.transition(self._state, reward = reward, discount=1.0)

        # if self._episode_ended or self._state >= 21:
        #     reward = self._state - 21 if self._state <= 21 else -21
        #     return ts.termination(np.array([self._state], dtype=np.int32), reward)
        # else:
        #     return ts.transition(
        #             np.array([self._state], dtype=np.int32), reward=0.0, discount=1.0)

def main():
    env = CogEnvDecoder(env_name="win_v1/RealGame.exe", no_graphics=False, time_scale=1, worker_id=1)
    action = [0.5, 0.5, 0.1, 0]
    env = Environment(Cog_env= env)
    env.reset()
    step = env.step(action=action)
    pass 

if __name__ == "__main__":
    main()




