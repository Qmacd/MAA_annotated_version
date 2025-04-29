"""
===============================================================================
文件名: gca_base.py

功能概述:
本文件定义了 GCA（Generative Competitive Adversarial）框架的虚基类 `GCABase`。
该类作为所有 GCA 模型的基类，提供了一些核心方法接口，所有子类需要实现这些方法以完成特定的功能。
`GCABase` 主要用于管理模型训练的超参数、设备设置、数据处理、模型初始化、训练和评估等关键任务。

主要功能:
1. **超参数初始化**:
   - 构造函数 `__init__` 初始化训练过程所需的超参数，包括生成器和判别器的数量、批量大小、训练轮数、学习率、输出路径、模型检查点路径等。
   - 支持设备选择，能够设置 CPU 或 GPU 作为训练设备。

2. **数据预处理与模型初始化**:
   - `process_data()`: 数据预处理接口，负责加载、清洗和划分数据。具体实现需在子类中定义。
   - `init_model()`: 初始化模型结构的接口。所有生成器和判别器的初始化都将通过该方法完成。
   - `init_dataloader()`: 用于初始化训练和测试数据加载器。
   - `init_hyperparameters()`: 用于初始化训练过程中所需的超参数，如学习率、优化器等。

3. **训练与评估**:
   - `train()`: 用于执行模型的训练过程，训练生成器和判别器。
   - `save_models()`: 用于保存训练过程中生成的模型。
   - `distill()`: 执行知识蒸馏过程，将知识从一个模型转移到另一个模型。
   - `visualize_and_evaluate()`: 用于评估模型性能，并将结果可视化，以便更好地理解和分析训练效果。

4. **随机种子设置**:
   - `set_seed()`: 用于设置随机种子，确保实验的可重复性。

使用方式:
- 该文件是 GCA 模型的基类，不能直接使用。用户需要继承 `GCABase` 类并实现其中的抽象方法，才能完成具体的模型训练与评估任务。
- 子类需要实现 `process_data()`、`init_model()`、`train()` 等方法，以满足具体实验需求。

===============================================================================
"""


from abc import ABC, abstractmethod
import random, torch, numpy as np
from utils.util import setup_device
import os


class GCABase(ABC):
    """
    GCA 框架的虚基类，定义核心方法接口。
    所有子类必须实现以下方法。
    """

    def __init__(self, N_pairs, batch_size, num_epochs,
                 generator_names, discriminators_names,
                 ckpt_dir, output_dir,
                 initial_learning_rate = 2e-4,
                 train_split = 0.8,
                 precise = torch.float32,
                 do_distill_epochs: int = 1,
                 cross_finetune_epochs: int = 5,
                 device = None,
                 seed=None,
                 ckpt_path="auto",):
        """
        初始化必备的超参数。

        :param N_pairs: 生成器or对抗器的个数
        :param batch_size: 小批次处理
        :param num_epochs: 预定训练轮数
        :param initial_learning_rate: 初始学习率
        :param generators: 建议是一个iterable object，包括了表示具有不同特征的生成器
        :param discriminators: 建议是一个iterable object，可以是相同的判别器
        :param ckpt_path: 各模型检查点
        :param output_path: 可视化、损失函数的log等输出路径
        """

        self.N = N_pairs
        self.initial_learning_rate = initial_learning_rate
        self.generator_names = generator_names
        self.discriminators_names = discriminators_names
        self.ckpt_dir = ckpt_dir
        self.ckpt_path = ckpt_path
        self.output_dir = output_dir
        self.batch_size = batch_size
        self.num_epochs = num_epochs
        self.train_split = train_split
        self.seed = seed
        self.do_distill_epochs = do_distill_epochs
        self.cross_finetune_epochs = cross_finetune_epochs
        self.device = device
        self.precise = precise

        self.set_seed(self.seed)  # 初始化随机种子
        self.device = setup_device(device)
        print("Running Device:", self.device)

        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)
            print("Output directory created! ")

        if not os.path.exists(self.ckpt_dir):
            os.makedirs(self.ckpt_dir)
            print("Checkpoint directory created! ")

    def set_seed(self, seed):
        """
        设置随机种子以确保实验的可重复性。

        :param seed: 随机种子
        """
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

    @abstractmethod
    def process_data(self):
        """数据预处理，包括读取、清洗、划分等"""
        pass

    @abstractmethod
    def init_model(self):
        """模型结构初始化"""
        pass

    @abstractmethod
    def init_dataloader(self):
        """初始化用于训练与评估的数据加载器"""
        pass

    @abstractmethod
    def init_hyperparameters(self):
        """初始化训练所需的超参数"""
        pass

    @abstractmethod
    def train(self):
        """执行训练过程"""
        pass

    @abstractmethod
    def save_models(self):
        """执行训练过程"""
        pass

    @abstractmethod
    def distill(self):
        """执行知识蒸馏过程"""
        pass

    @abstractmethod
    def visualize_and_evaluate(self):
        """评估模型性能并可视化结果"""
        pass

    @abstractmethod
    def init_history(self):
        """初始化训练过程中的指标记录结构"""
        pass
