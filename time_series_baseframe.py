"""
===============================================================================
文件名: base_time_series.py

功能概述:
本文件定义了一个基于时间序列数据的 GCA模型框架。主要功能包括数据处理、模型初始化、训练、预测、模型评估与保存等。模型使用生成器和判别器结构，支持时间序列的生成与分类任务。

主要功能:
1. **数据处理**: 处理输入的时间序列数据，包括数据加载、特征选择、标签生成、训练集和测试集划分、归一化等。通过滑动窗口方法生成输入输出数据序列。
2. **模型结构**: 定义并初始化生成器（Generators）和判别器（Discriminators）的结构。支持多种生成器和判别器类型，且模型的配置可根据需要进行扩展。
3. **训练与评估**: 使用训练数据训练模型，并通过自定义的 `train_baseframe` 函数评估模型的性能。支持多轮训练、超参数优化等。
4. **自动化检查点管理**: 支持自动加载和保存模型的检查点，并能够根据时间戳获取最新的模型检查点进行预测。
5. **预测与可视化**: 加载训练好的生成器模型进行预测，并对结果进行可视化评估。支持生成分类标签并对预测结果进行详细评估。
6. **超参数初始化**: 初始化训练所需的超参数，设置学习率、优化器参数、权重矩阵等。

配置与参数:
- 用户通过命令行传递各种参数，如数据路径、生成器数量、训练轮数、学习率、设备选择等。
- 提供模型训练过程中的日志记录和时间统计功能，帮助用户跟踪模型的训练进度和性能。

使用方式:
- 通过实例化 `base_time_series` 类并传入相关参数来初始化和训练模型。
- 提供自动加载检查点功能，并能够通过预测函数进行模型推理。

===============================================================================
"""


from GCA_base import GCABase
import torch
import numpy as np
from functools import wraps
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import TensorDataset, DataLoader
from utils.baseframe_trainer import train_baseframe
from typing import List, Optional
import models
import os
import time
import glob
from utils.evaluate_visualization import evaluate_best_models_vwap

def log_execution_time(func):
    """装饰器：记录函数的运行时间，并动态获取函数名"""
    @wraps(func)  # 保留原函数的元信息（如 __name__）
    def wrapper(*args, **kwargs):
        start_time = time.time()  # 记录开始时间
        result = func(*args, **kwargs)  # 执行目标函数
        end_time = time.time()  # 记录结束时间
        elapsed_time = end_time - start_time  # 计算耗时
        # 动态获取函数名（支持类方法和普通函数）
        func_name = func.__name__
        print(f"GCA_time_series - '{func_name}' elapse time: {elapsed_time:.4f} sec")
        return result
    return wrapper


def generate_labels(y):
    """
    根据每个时间步 y 是否比前一时刻更高，生成三分类标签：
      - 2: 当前值 > 前一时刻（上升）
      - 0: 当前值 < 前一时刻（下降）
      - 1: 当前值 == 前一时刻（平稳）
    对于第一个时间步，默认赋值为1（平稳）。

    参数：
        y: 数组，形状为 (样本数, ) 或 (样本数, 1)
    返回：
        labels: 生成的标签数组，长度与 y 相同
    """
    y = np.array(y).flatten()  # 转成一维数组
    labels = [0]  # 对于第一个样本，默认平稳
    for i in range(1, len(y)):
        if y[i] > y[i - 1]:
            labels.append(2)
        elif y[i] < y[i - 1]:
            labels.append(0)
        else:
            labels.append(1)
    return np.array(labels)


class base_time_series(GCABase):
    def __init__(self, args, N_pairs: int, batch_size: int, num_epochs: int,
                 generators_names: List, discriminators_names: Optional[List],
                 ckpt_dir: str, output_dir: str,
                 window_sizes: int,
                 initial_learning_rate: float = 2e-5,
                 train_split: float = 0.8,
                 do_distill_epochs: int = 1,
                 cross_finetune_epochs: int = 5,
                 precise=torch.float32,
                 device=None,
                 seed: int = None,
                 ckpt_path: str = None,
                 gan_weights=None,
                 ):
        """
        初始化必备的超参数。

        :param N_pairs: 生成器or对抗器的个数
        :param batch_size: 小批次处理
        :param num_epochs: 预定训练轮数
        :param initial_learning_rate: 初始学习率
        :param generators_names: list object，包括了表示具有不同特征的生成器的名称
        :param discriminators_names: list object，包括了表示具有不同判别器的名称，如果没有就不写默认一致
        :param ckpt_dir: 各模型检查点保存目录
        :param output_path: 可视化、损失函数的log等输出目录
        :param ckpt_path: 预测时保存的检查点
        """
        super().__init__(N_pairs, batch_size, num_epochs,
                         generators_names, discriminators_names,
                         ckpt_dir, output_dir,
                         initial_learning_rate,
                         train_split,
                         precise,
                         do_distill_epochs, cross_finetune_epochs,
                         device,
                         seed,
                         ckpt_path)  # 调用父类初始化

        self.args = args
        self.window_sizes = window_sizes
        # 初始化空字典
        self.generator_dict = {}
        self.discriminator_dict = {"default": models.Discriminator3}

        # 遍历 model 模块下的所有属性
        for name in dir(models):
            obj = getattr(models, name)
            if isinstance(obj, type) and issubclass(obj, torch.nn.Module):
                lname = name.lower()
                if "generator" in lname:
                    key = lname.replace("generator_", "")
                    self.generator_dict[key] = obj
                elif "discriminator" in lname:
                    key = lname.replace("discriminator", "")
                    self.discriminator_dict[key] = obj

        self.gan_weights = gan_weights

        self.init_hyperparameters()

    @log_execution_time  # 语法糖，记录函数运行时间
    def process_data(self, data_path, start_row, end_row,  target_columns, feature_columns):
        """
        处理输入数据，包括加载、拆分和归一化。

        参数:
            data_path (str): CSV 数据文件的路径
            target_columns (list): 目标列的索引
            feature_columns (list): 特征列的索引

        返回:
            tuple: (train_x, test_x, train_y, test_y, y_scaler)
        """

        print(f"Processing data with seed: {self.seed}")  # 打印随机种子
        # 载入数据
        data = pd.read_csv(data_path)
        print(f'dataset name: {data_path}')
        # 切片目标数据
        y = data.iloc[start_row:end_row, target_columns].values
        target_column_names = data.columns[target_columns]
        print("Target columns:", target_column_names)

        # 切片特征
        x = data.iloc[start_row:end_row, feature_columns].values
        feature_column_names = data.columns[feature_columns]
        print("Feature columns:", feature_column_names)

        # 切片训练集和测试集，3：7
        train_size = int(data.iloc[start_row:end_row].shape[0] * self.train_split)
        train_x, test_x = x[:train_size], x[train_size:]
        train_y, test_y = y[:train_size], y[train_size:]

        #  归一化
        """
        fit_transform 方法会计算 train_y 的最小值和最大值，并将这些信息存储在 self.y_scaler 中。
        然后，它会将 train_y 缩放到 [0, 1] 的范围，并将结果赋值给 self.train_y。
        """
        self.x_scaler = MinMaxScaler(feature_range=(0, 1))  # 保存归一化方法
        self.y_scaler = MinMaxScaler(feature_range=(0, 1))  # 保存归一化方法
        self.train_x = self.x_scaler.fit_transform(train_x)
        self.test_x = self.x_scaler.transform(test_x)
        self.train_y = self.y_scaler.fit_transform(train_y)
        self.test_y = self.y_scaler.transform(test_y)

        self.train_labels = generate_labels(self.train_y)  # 生成训练集的分类标签（直接在 GPU 上生成）
        self.test_labels = generate_labels(self.test_y)   # 生成测试集的分类标签
        print(self.train_y[:5])
        print(self.train_labels[:5])

    def create_sequences_combine(self, x, y, label, window_size, start):

        # 初始化
        x_ = []
        y_ = []
        y_gan = []
        label_gan = []

        # 利用窗口滑动处理数据
        # 窗口大小为 window_size，从 start 开始滑动
        for i in range(start, x.shape[0]):
            tmp_x = x[i - window_size: i, :]
            tmp_y = y[i]
            tmp_y_gan = y[i - window_size: i + 1]
            tmp_label_gan = label[i - window_size: i + 1]

            x_.append(tmp_x)
            y_.append(tmp_y)
            y_gan.append(tmp_y_gan)
            label_gan.append(tmp_label_gan)

        # 数据转成 tensor
        x_ = torch.from_numpy(np.array(x_)).float()
        y_ = torch.from_numpy(np.array(y_)).float()
        y_gan = torch.from_numpy(np.array(y_gan)).float()
        label_gan = torch.from_numpy(np.array(label_gan)).float()
        return x_, y_, y_gan, label_gan

    @log_execution_time  # 语法糖，记录函数运行时间
    def init_dataloader(self):
        """初始化用于训练与评估的数据加载器"""

        # 利用create_sequences_combine构造训练集、测试集数据
        train_data_list = [
            self.create_sequences_combine(self.train_x, self.train_y, self.train_labels, w, self.window_sizes[-1])
            for w in self.window_sizes
        ]
        test_data_list = [
            self.create_sequences_combine(self.test_x, self.test_y, self.test_labels, w, self.window_sizes[-1])
            for w in self.window_sizes
        ]

        # 从 train_data_list 和 test_data_list 中提取输入特征、目标值、生成器训练所需的目标序列和标签数据，并将它们转移到指定的设备
        self.train_x_all = [x.to(self.device) for x, _, _, _ in train_data_list]
        self.train_y_all = train_data_list[0][1]
        self.train_y_gan_all = [y_gan.to(self.device) for _, _, y_gan, _ in train_data_list]
        self.train_label_gan_all = [label_gan.to(self.device) for _, _, _, label_gan in train_data_list]
        self.test_x_all = [x.to(self.device) for x, _, _, _ in test_data_list]
        self.test_y_all = test_data_list[0][1]
        self.test_y_gan_all = [y_gan.to(self.device) for _, _, y_gan, _ in test_data_list]
        self.test_label_gan_all = [label_gan.to(self.device) for _, _, _, label_gan in test_data_list]

        # 数据一致性检查，如果有任何不匹配的目标值，代码会抛出 AssertionError
        assert all(torch.equal(train_data_list[0][1], y) for _, y, _, _ in train_data_list), "Train y mismatch!"
        assert all(torch.equal(test_data_list[0][1], y) for _, y, _, _ in test_data_list), "Test y mismatch!"

        """
        train_x_all.shape  # (N, N, W, F)  不同 window_size 会导致 W 不一样，只能在 W 相同时用 stack
        train_y_all.shape  # (N,)
        train_y_gan_all.shape  # (3, N, W+1)
        """

        self.dataloaders = []
        # 遍历不同 window_size 的训练集和测试集，构造 DataLoader
        for i, (x, y_gan, label_gan) in enumerate(
                zip(self.train_x_all, self.train_y_gan_all, self.train_label_gan_all)):
            shuffle_flag = ("transformer" in self.generator_names[i])  # 最后一个设置为 shuffle=True，其余为 False

            dataloader = DataLoader(
                TensorDataset(x, y_gan, label_gan),
                batch_size=self.batch_size,
                shuffle=shuffle_flag,
                generator=torch.manual_seed(self.seed),
                drop_last=True  # 丢弃最后一个不足 batch size 的数据
            )

            self.dataloaders.append(dataloader)

    def init_model(self, num_cls=3):

        """模型结构初始化"""

        # 检查生成器和判别器一致性
        assert len(self.generator_names) == self.N, "Generators and Discriminators mismatch!"
        assert isinstance(self.generator_names, list)
        for i in range(self.N):
            assert isinstance(self.generator_names[i], str)

        # 初始化生成器和判别器的空列表
        self.generators = []
        self.discriminators = []

        # 遍历并初始化每个生成器和判别器
        for i, name in enumerate(self.generator_names):
            # 获取对应的 x, y
            x = self.train_x_all[i]
            y = self.train_y_all[i]

            # 初始化生成器
            GenClass = self.generator_dict[name]
            if "transformer" in name:
                gen_model = GenClass(x.shape[-1], output_len=y.shape[-1]).to(self.device)
            else:
                gen_model = GenClass(x.shape[-1], y.shape[-1]).to(self.device)

            self.generators.append(gen_model)

            # 初始化判别器（默认只用 Discriminator3）
            DisClass = self.discriminator_dict[
                "default" if self.discriminators_names is None else self.discriminators_names[i]]
            dis_model = DisClass(self.window_sizes[i], out_size=y.shape[-1], num_cls=3).to(self.device)
            self.discriminators.append(dis_model)

    def init_hyperparameters(self, ):
        """初始化训练所需的超参数"""

        # 初始化 init_GDweight 权重矩阵。对角线上为1，其余为0，最后一列为1.0
        self.init_GDweight = []
        for i in range(self.N):  # 遍历生成器和判别器数量
            row = [0.0] * self.N
            row[i] = 1.0
            row.append(1.0)  # 最后一列为 scale
            self.init_GDweight.append(row)  # 将每行的权重矩阵 row 添加到 self.init_GDweight 列表中。

        if self.gan_weights is None:
            # 最终：均分组合，最后一列为1.0
            final_row = [round(1.0 / self.N, 3)] * self.N + [1.0]
            self.final_GDweight = [final_row[:] for _ in range(self.N)]
        else:
            pass
        # 初始化学习率和优化器参数
        self.g_learning_rate = self.initial_learning_rate
        self.d_learning_rate = self.initial_learning_rate

        # 初始化 Adam 优化器的超参数
        self.adam_beta1, self.adam_beta2 = (0.9, 0.999)

        # 初始化学习率调度器的参数
        self.schedular_factor = 0.1
        self.schedular_patience = 16
        self.schedular_min_lr = 1e-7

    def train(self, logger):
        return train_baseframe(self.generators[0], self.dataloaders[0],
                                                    self.y_scaler, self.train_x_all[0], self.train_y_all, self.test_x_all[0],
                                                    self.test_y_all,
                                                    self.num_epochs,
                                                    self.output_dir,
                                                    self.device,
                                                    logger=logger)

    def save_models(self, best_model_state):
        """
        保存所有 generator 和 discriminator 的模型参数，包含时间戳、模型名称或编号。
        """

        timestamp = time.strftime("%Y%m%d_%H%M%S")
        ckpt_dir = os.path.join(self.ckpt_dir, timestamp)
        gen_dir = os.path.join(ckpt_dir, "generators")
        disc_dir = os.path.join(ckpt_dir, "discriminators")
        os.makedirs(gen_dir, exist_ok=True)
        os.makedirs(disc_dir, exist_ok=True)

        for i, gen in enumerate(self.generators):
            gen_name = type(gen).__name__
            save_path = os.path.join(gen_dir, f"{i + 1}_{gen_name}.pt")
            torch.save(gen.state_dict(), save_path)

        for i, disc in enumerate(self.discriminators):
            disc_name = type(disc).__name__
            save_path = os.path.join(disc_dir, f"{i + 1}_{disc_name}.pt")
            torch.save(disc.state_dict(), save_path)

        print("All models saved with timestamp and identifier.")

    # 取指定目录中最新的检查点（checkpoint）文件夹。
    def get_latest_ckpt_folder(self):
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        all_subdirs = [d for d in glob.glob(os.path.join(self.ckpt_dir, timestamp[0] + "*")) if os.path.isdir(d)]
        if not all_subdirs:
            raise FileNotFoundError("❌ No checkpoint records!!")
        latest = max(all_subdirs, key=os.path.getmtime)
        print(f"📂 Auto loaded checkpoint file: {latest}")
        return latest

    # 装载模型
    def load_model(self):
        gen_path = os.path.join(self.ckpt_path, "g{gru}", "generator.pt")
        if os.path.exists(gen_path):
            self.generators[0].load_state_dict(torch.load(gen_path, map_location=self.device))
            print(f"✅ Loaded generator from {gen_path}")
        else:
            raise FileNotFoundError(f"❌ Generator checkpoint not found at: {gen_path}")


    def pred(self):
        """
        如果 ckpt_path 是 "auto"，则自动获取最新的检查点文件夹路径。
        加载所有生成器的最佳模型权重。
        使用加载的生成器模型对训练和测试数据进行预测。
        调用 evaluate_best_models_vwap() 函数来评估这些模型并返回结果。
        """
        if self.ckpt_path == "auto":
            self.ckpt_path = self.get_latest_ckpt_folder()

        print("Start predicting with all generators..")
        best_model_state = [None for _ in range(self.N)]
        current_path = os.path.join(self.ckpt_path, "generators")

        for i, gen in enumerate(self.generators):
            gen_name = type(gen).__name__
            save_path = os.path.join(current_path, f"{i + 1}_{gen_name}.pt")
            best_model_state[i] = self.generator_dict[self.generator_names[i]].load_state_dict(torch.load(save_path))

        results = evaluate_best_models_vwap(self.generators, best_model_state, self.train_x_all, self.train_y_all,
                                       self.test_x_all, self.test_y_all, self.y_scaler,
                                       self.output_dir)
        return results

    def distill(self):
        """评估模型性能并可视化结果"""
        pass

    def visualize_and_evaluate(self):
        """评估模型性能并可视化结果"""
        pass

    def init_history(self):
        """初始化训练过程中的指标记录结构"""
        pass
