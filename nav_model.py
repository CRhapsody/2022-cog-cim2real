from typing import Callable, Dict, List, Optional, Tuple, Type, Union

import gym
import torch as th
from torch import nn
from stable_baselines3.common.policies import ActorCriticPolicy
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from Cogenvdecoder.CogEnvDecoder import CogEnvDecoder
from nav_gym_env import NavEnvironment
from stable_baselines3 import PPO, DDPG, A2C, TD3


class NavCustomCombinedExtractor(BaseFeaturesExtractor):
    def __init__(self, observation_space: gym.spaces.Dict):
        # We do not know features-dim here before going over all the items,
        # so put something dummy for now. PyTorch requires calling
        # nn.Module.__init__ before adding modules
        super(NavCustomCombinedExtractor, self).__init__(observation_space, features_dim=1)

        extractors = {}

        total_concat_size = 0
        # We need to know size of the output of this extractor,
        # so go over all the spaces and compute output feature sizes
        for key, subspace in observation_space.spaces.items():
            if key == "laser":
                # We will just downsample one channel of the image by 4x4 and flatten.
                # Assume the image is single-channel (subspace.shape[0] == 0)
                extractors[key] = nn.Sequential(
                    nn.Conv1d(1, 1, kernel_size=3),
                    nn.MaxPool1d(kernel_size=3),
                    nn.Tanh(),
                    nn.Conv1d(1, 1, kernel_size=3),
                    nn.Tanh(),
                    nn.Flatten(),
                    nn.Linear(17, 16)
                )
                # extractors[key] = nn.Sequential(nn.MaxPool2d(4), nn.Flatten())
                total_concat_size += 16
            elif key == "vec":
                # Run through a simple MLP
                # extractors[key] = nn.Linear(subspace.shape[0], 16)
                # total_concat_size += 16
                extractors[key] = nn.Identity()
                total_concat_size += subspace.shape[0]
            elif key == "image":
                extractors[key] = nn.Sequential(
                    nn.Conv2d(3, 32, kernel_size=3, padding=0),
                    nn.ReLU(),
                    nn.Conv2d(32, 64, kernel_size=3, padding=0),
                    nn.ReLU(),
                    nn.Flatten(),
                    nn.Linear(230400, 128),
                    nn.Tanh(),
                    nn.Linear(128, 32)
                )
                total_concat_size += 32 # TODO

        self.extractors = nn.ModuleDict(extractors)

        # Update the features dim manually
        self._features_dim = total_concat_size

    def forward(self, observations) -> th.Tensor:
        encoded_tensor_list = []

        # self.extractors contain nn.Modules that do all the processing.
        for key, extractor in self.extractors.items():
            encoded_tensor_list.append(extractor(observations[key]))
        # Return a (B, self._features_dim) PyTorch tensor, where B is batch dimension.
        return th.cat(encoded_tensor_list, dim=1)


class NavNetwork(nn.Module):
    """
    Custom network for policy and value function.
    It receives as input the features extracted by the feature extractor.

    :param feature_dim: dimension of the features extracted with the features_extractor (e.g. features from a CNN)
    :param last_layer_dim_pi: (int) number of units for the last layer of the policy network
    :param last_layer_dim_vf: (int) number of units for the last layer of the value network
    """

    def __init__(
        self,
        feature_dim: int,
        last_layer_dim_pi: int = 64,
        last_layer_dim_vf: int = 64,
    ):
        super(NavNetwork, self).__init__()

        # IMPORTANT:
        # Save output dimensions, used to create the distributions
        self.latent_dim_pi = last_layer_dim_pi
        self.latent_dim_vf = last_layer_dim_vf

        # self.shared_net = nn.Sequential(
            # nn.Linear(feature_dim, 64), nn.Tanh(),
            # nn.Linear(128, 128), nn.ReLU()
        # )

        # Policy network
        self.policy_net = nn.Sequential(
            nn.Linear(feature_dim, 64), nn.Tanh(),
            nn.Linear(64, last_layer_dim_pi), nn.Tanh()
        )
        # Value network
        self.value_net = nn.Sequential(
            nn.Linear(feature_dim, 64), nn.Tanh(),
            nn.Linear(64, last_layer_dim_vf), nn.Tanh()
        )

    def forward(self, features: th.Tensor) -> Tuple[th.Tensor, th.Tensor]:
        """
        :return: (th.Tensor, th.Tensor) latent_policy, latent_value of the specified network.
            If all layers are shared, then ``latent_policy == latent_value``
        """
        pi = self.policy_net(features)
        vl = self.value_net(features)
        return pi, vl

    def forward_actor(self, features: th.Tensor) -> th.Tensor:
        pi = self.policy_net(features)
        return pi

    def forward_critic(self, features: th.Tensor) -> th.Tensor:
        vl = self.value_net(features)
        return vl


class NavActorCriticPolicy(ActorCriticPolicy):
    def __init__(
        self,
        observation_space: gym.spaces.Space,
        action_space: gym.spaces.Space,
        lr_schedule: Callable[[float], float],
        net_arch: Optional[List[Union[int, Dict[str, List[int]]]]] = None,
        activation_fn: Type[nn.Module] = nn.Tanh,
        *args,
        **kwargs,
    ):

        super(NavActorCriticPolicy, self).__init__(
            observation_space,
            action_space,
            lr_schedule,
            net_arch,
            activation_fn,
            features_extractor_class=NavCustomCombinedExtractor,
            # Pass remaining arguments to base class
            *args,
            **kwargs,
        )
        # Disable orthogonal initialization
        self.ortho_init = False

    def _build_mlp_extractor(self) -> None:
        self.mlp_extractor = NavNetwork(self.features_dim)


def test():
    cog_env = CogEnvDecoder(
        env_name="C:\\Users\\Administrator\\PycharmProjects\\2022-cog-cim2real\\win_v2.1\\cog_sim2real_env.exe",
        no_graphics=False, time_scale=3, worker_id=1)
    env = NavEnvironment(Cog_env=cog_env, env_id=1)
    model = PPO(NavActorCriticPolicy, env, verbose=1)
    model.learn(5000)


if __name__ == '__main__':
    test()

# model = PPO(CustomActorCriticPolicy, "CartPole-v1", verbose=1)
# model.learn(5000)