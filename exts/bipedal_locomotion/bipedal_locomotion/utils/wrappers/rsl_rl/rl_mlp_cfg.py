# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from dataclasses import MISSING
from typing import Literal

from isaaclab.utils import configclass
from isaaclab_rl.rsl_rl import RslRlPpoAlgorithmCfg


@configclass
class RslRlPpoAlgorithmMlpCfg(RslRlPpoAlgorithmCfg):
    """支持MLP的PPO算法配置类 - 扩展基础PPO配置以支持多层感知机 / PPO algorithm configuration with MLP support - extends base PPO config for multi-layer perceptron support"""

    # runner_type: str = "OnPolicyRunner"

    obs_history_len: int = 1  # 观测历史长度 - 控制输入序列的时间步数 / Observation history length - controls timesteps in input sequence


@configclass
class EncoderCfg:
    """编码器配置类 - 用于处理复杂观测输入的神经网络编码器 / Encoder configuration class - neural network encoder for processing complex observation inputs"""
    
    output_detach: bool = True                    # 输出分离 - 阻止梯度回传到编码器 / Output detach - prevents gradient backpropagation to encoder
    num_input_dim: int = MISSING                  # 输入维度 - 必须在使用时指定 / Input dimensions - must be specified when used
    num_output_dim: int = 3                       # 输出维度 - 编码后的特征维度 / Output dimensions - encoded feature dimensions
    hidden_dims: list[int] = [256, 128]           # 隐藏层维度列表 - 定义网络架构 / Hidden layer dimensions - defines network architecture
    activation: str = "elu"                       # 激活函数 - ELU激活函数 / Activation function - ELU activation
    orthogonal_init: bool = False                 # 正交初始化 - 是否使用正交权重初始化 / Orthogonal initialization - whether to use orthogonal weight init


import os
import copy
import torch
def export_mlp_as_onnx(mlp, path, name, input_dim):

    """将MLP模型导出为ONNX格式 - 用于部署到不同平台 / Export MLP model to ONNX format - for deployment to different platforms
    Args:
        mlp: 要导出的MLP模型 / MLP model to export
        path: 导出路径 / Export path
        name: 模型文件名 / Model filename
        input_dim: 输入维度 / Input dimensions
    """
    os.makedirs(path, exist_ok=True)
    path = os.path.join(path, name + ".onnx")
    model = copy.deepcopy(mlp).to("cpu")
    model.eval()

    dummy_input = torch.randn(input_dim)
    input_names = ["mlp_input"]
    output_names = ["mlp_output"]

    torch.onnx.export(
        model,
        dummy_input,
        path,
        verbose=True,
        input_names=input_names,
        output_names=output_names,
        export_params=True,
        opset_version=13,
    )
    print("Exported policy as onnx script to: ", path)

def export_policy_as_jit(actor_critic, path):
    """将策略导出为TorchScript JIT格式 - 用于C++部署 / Export policy to TorchScript JIT format - for C++ deployment
    
    Args:
        actor_critic: Actor-Critic模型 / Actor-Critic model
        path: 导出路径 / Export path
    """
    os.makedirs(path, exist_ok=True)
    path = os.path.join(path, "policy.pt")
    model = copy.deepcopy(actor_critic.actor).to("cpu")
    traced_script_module = torch.jit.script(model)
    traced_script_module.save(path)
