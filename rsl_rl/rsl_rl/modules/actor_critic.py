# SPDX-FileCopyrightText: Copyright (c) 2021 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this
# list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
# this list of conditions and the following disclaimer in the documentation
# and/or other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its
# contributors may be used to endorse or promote products derived from
# this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
#
# Copyright (c) 2021 ETH Zurich, Nikita Rudin

import numpy as np

import torch
import torch.nn as nn
from torch.distributions import Normal
from torch.nn.modules import rnn


class ActorCritic(nn.Module):
    """Actor-Critic网络架构 - 同时包含策略网络和价值网络 / Actor-Critic network architecture - contains both policy and value networks"""
    
    is_recurrent = False    # 不使用循环神经网络 / Not using recurrent neural networks
    is_sequence = False     # 不处理序列数据 / Not processing sequence data
    is_vae = False          # 不是变分自编码器 / Not a variational autoencoder

    def __init__(
        self,
        num_actor_obs,                      # Actor观测维度 / Actor observation dimensions
        num_critic_obs,                     # Critic观测维度 / Critic observation dimensions
        num_actions,                        # 动作维度 / Action dimensions
        actor_hidden_dims=[256, 256, 256],  # Actor隐藏层维度 / Actor hidden layer dimensions
        critic_hidden_dims=[256, 256, 256], # Critic隐藏层维度 / Critic hidden layer dimensions
        activation="elu",                   # 激活函数类型 / Activation function type
        orthogonal_init=False,              # 是否使用正交初始化 / Whether to use orthogonal initialization
        init_noise_std=1.0,                 # 初始噪声标准差 / Initial noise standard deviation
        **kwargs,
    ):
        """初始化Actor-Critic网络 / Initialize Actor-Critic network"""
        if kwargs:
            print(
                "ActorCritic.__init__ got unexpected arguments, which will be ignored: "
                + str([key for key in kwargs.keys()])
            )
        super(ActorCritic, self).__init__()

        self.orthogonal_init = orthogonal_init
        self.num_actor_obs = num_actor_obs
        self.num_critic_obs = num_critic_obs

        activation = get_activation(activation)

        # 策略网络 (Actor) 构建 / Policy network (Actor) construction
        actor_layers = []
        actor_layers.append(nn.Linear(num_actor_obs, actor_hidden_dims[0]))
        if self.orthogonal_init:
            torch.nn.init.orthogonal_(actor_layers[-1].weight, np.sqrt(2))
        actor_layers.append(activation)
        for l in range(len(actor_hidden_dims)):
            if l == len(actor_hidden_dims) - 1:
                actor_layers.append(nn.Linear(actor_hidden_dims[l], num_actions))
                if self.orthogonal_init:
                    torch.nn.init.orthogonal_(actor_layers[-1].weight, 0.01)
                    torch.nn.init.constant_(actor_layers[-1].bias, 0.0)
            else:
                actor_layers.append(
                    nn.Linear(actor_hidden_dims[l], actor_hidden_dims[l + 1])
                )
                if self.orthogonal_init:
                    torch.nn.init.orthogonal_(actor_layers[-1].weight, np.sqrt(2))
                    torch.nn.init.constant_(actor_layers[-1].bias, 0.0)
                actor_layers.append(activation)
                # 可选：添加层归一化 / Optional: add layer normalization
                # actor_layers.append(torch.nn.LayerNorm(actor_hidden_dims[l + 1]))
        self.actor = nn.Sequential(*actor_layers)

        # 价值网络 (Critic) 构建 / Value network (Critic) construction
        critic_layers = []
        critic_layers.append(nn.Linear(num_critic_obs, critic_hidden_dims[0]))
        critic_layers.append(activation)
        for l in range(len(critic_hidden_dims)):
            if l == len(critic_hidden_dims) - 1:
                # 输出层：单一价值输出 / Output layer: single value output
                critic_layers.append(nn.Linear(critic_hidden_dims[l], 1))
                if self.orthogonal_init:
                    torch.nn.init.orthogonal_(critic_layers[-1].weight, 0.01)
                    torch.nn.init.constant_(critic_layers[-1].bias, 0.0)
            else:
                # 隐藏层 / Hidden layer
                critic_layers.append(
                    nn.Linear(critic_hidden_dims[l], critic_hidden_dims[l + 1])
                )
                if self.orthogonal_init:
                    torch.nn.init.orthogonal_(critic_layers[-1].weight, np.sqrt(2))
                    torch.nn.init.constant_(critic_layers[-1].bias, 0.0)
                critic_layers.append(activation)
                # 可选：添加层归一化 / Optional: add layer normalization
                # critic_layers.append(torch.nn.LayerNorm(critic_hidden_dims[l + 1]))

        self.critic = nn.Sequential(*critic_layers)

        print(f"Actor MLP: {self.actor}")
        print(f"Critic MLP: {self.critic}")

        # 动作噪声参数 / Action noise parameters
        # self.std = nn.Parameter(init_noise_std * torch.ones(num_actions))  # 固定标准差方法 / Fixed std method
        self.logstd = nn.Parameter(torch.zeros(num_actions))  # 可学习的对数标准差 / Learnable log standard deviation
        self.distribution = None  # 当前动作分布 / Current action distribution
        
        # 禁用参数验证以加速 / Disable args validation for speedup
        Normal.set_default_validate_args = False


    @staticmethod
    # not used at the moment
    def init_weights(sequential, scales):
        """初始化权重的静态方法（当前未使用）/ Static method for weight initialization (currently unused)"""
        [
            torch.nn.init.orthogonal_(module.weight, gain=scales[idx])
            for idx, module in enumerate(
                mod for mod in sequential if isinstance(mod, nn.Linear)
            )
        ]

    def reset(self, dones=None):
        pass

    def forward(self):
        raise NotImplementedError

    @property
    def action_mean(self):
        return self.distribution.mean

    @property
    def action_std(self):
        return self.distribution.stddev

    @property
    def entropy(self):
        return self.distribution.entropy().sum(dim=-1)

    def update_distribution(self, observations):
        """更新动作分布 / Update action distribution
        
        Args:
            observations: 输入观测 / Input observations
        """
        mean = self.actor(observations)
        self.distribution = Normal(mean, mean * 0.0 + torch.exp(self.logstd))

    def act(self, observations, **kwargs):
        """执行动作采样 / Perform action sampling
        
        Args:
            observations: 输入观测 / Input observations
            
        Returns:
            采样的动作 / Sampled actions
        """
        self.update_distribution(observations)
        return self.distribution.sample()

    def get_actions_log_prob(self, actions):
        """获取动作的对数概率 / Get log probability of actions
        
        Args:
            actions: 输入动作 / Input actions
            
        Returns:
            动作的对数概率 / Log probability of actions
        """
        return self.distribution.log_prob(actions).sum(dim=-1)

    def act_inference(self, observations):
        """推理时的确定性动作选择 / Deterministic action selection for inference
        
        Args:
            observations: 输入观测 / Input observations
            
        Returns:
            确定性动作（均值）/ Deterministic actions (mean)
        """
        actions_mean = self.actor(observations)
        return actions_mean

    def evaluate(self, critic_observations, **kwargs):
        """评估状态价值 / Evaluate state value
        
        Args:
            critic_observations: 评价器观测 / Critic observations
            
        Returns:
            状态价值估计 / State value estimate
        """
        value = self.critic(critic_observations)
        return value


def get_activation(act_name):
    """根据名称获取激活函数 / Get activation function by name
    
    Args:
        act_name: 激活函数名称 / Activation function name
        
    Returns:
        对应的激活函数 / Corresponding activation function
    """
    if act_name == "elu":
        return nn.ELU()           # 指数线性单元 / Exponential Linear Unit
    elif act_name == "selu":
        return nn.SELU()          # 缩放指数线性单元 / Scaled Exponential Linear Unit
    elif act_name == "relu":
        return nn.ReLU()          # 修正线性单元 / Rectified Linear Unit
    elif act_name == "crelu":
        return nn.ReLU()          # 连接修正线性单元 / Concatenated ReLU
    elif act_name == "lrelu":
        return nn.LeakyReLU()     # 泄漏修正线性单元 / Leaky Rectified Linear Unit
    elif act_name == "tanh":
        return nn.Tanh()          # 双曲正切 / Hyperbolic Tangent
    elif act_name == "sigmoid":
        return nn.Sigmoid()       # Sigmoid函数 / Sigmoid function
    else:
        print("invalid activation function!")
        return None