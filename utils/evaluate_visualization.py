"""
该代码实现了一个用于评估生成对抗网络（GANs）模型的模块，包含多个功能：
1. **验证模型性能**：通过计算均方误差（MSE）和分类准确率（acc）来评估模型的表现。
2. **损失曲线可视化**：提供多个函数来绘制生成器（Generator）和判别器（Discriminator）的损失变化曲线，帮助分析训练过程中的损失表现。
3. **拟合曲线绘制**：绘制模型预测值与真实值之间的拟合曲线，评估模型在训练和测试集上的拟合效果。
4. **评估指标保存**：将每个生成器模型的训练和测试指标（如MSE、MAE、RMSE、MAPE）保存到文本文件中，方便后续查看。
5. **CSV文件输出**：将模型的预测结果与真实标签输出到CSV文件中，便于进一步分析和可视化。

### 函数说明：
- `validate()`：验证模型在验证集上的表现，计算MSE损失和分类准确率。
- `plot_generator_losses()`：绘制各生成器（G1, G2, G3）的训练过程中损失曲线。
- `plot_discriminator_losses()`：绘制各判别器（D1, D2, D3）的损失曲线。
- `visualize_overall_loss()`：绘制生成器和判别器的总损失变化曲线。
- `plot_mse_loss()`：绘制生成器的训练和验证集上的MSE损失曲线。
- `inverse_transform()`：使用指定的y_scaler进行逆转换，将预测值还原为原始尺度。
- `compute_metrics()`：计算回归问题中的MSE、MAE、RMSE、MAPE等评估指标。
- `plot_fitting_curve()`：绘制真实值和预测值之间的拟合曲线，并保存结果图像。
- `save_metrics()`：将MSE、MAE、RMSE、MAPE等评估指标保存为文本文件。
- `evaluate_best_models_vwap()`：评估生成器模型（VWAP、TWAP任务），计算训练和测试集的各种指标，并保存测试集的预测结果和真实标签到CSV文件。

### 使用说明：
- 该代码主要针对生成对抗网络（GAN）模型进行评估，适用于包含多个生成器和判别器的场景。
- 评估过程中，计算了训练和验证集的损失，保存了每个模型的训练过程，并提供了可视化图形，以便研究和分析模型性能。
- 预测结果会被保存到指定的文件路径下，文件格式为CSV，以便后续进行进一步分析。
- 结果还包括不同模型（生成器和判别器）的性能指标（如MSE、MAE、RMSE、MAPE等）以及可视化图形。

"""

import torch.nn.functional as F
import os
from sklearn.metrics import mean_squared_error, mean_absolute_error
import numpy as np
import torch
import matplotlib.pyplot as plt
import logging
import pandas as pd

def validate(model, val_x, val_y, val_labels):
    """
    验证模型的性能。

    Args:
        model (nn.Module): 模型。
        val_x (torch.Tensor): 验证集的输入。
        val_y (torch.Tensor): 验证集的标签。
        val_labels (torch.Tensor): 验证集的标签。

    Returns:
        mse_loss (float): 均方误差。
        acc (float): 分类准确率。
    """
    model.eval()  # 将模型设置为评估模式
    with torch.no_grad():  # 禁止计算梯度
        val_x = val_x.clone().detach().float()

        # 检查val_y的类型，如果是numpy.ndarray则转换为torch.Tensor
        if isinstance(val_y, np.ndarray):
            val_y = torch.tensor(val_y).float()
        else:
            val_y = val_y.clone().detach().float()

        # labels 用于分类
        if isinstance(val_labels, np.ndarray):
            val_lbl_t = torch.tensor(val_labels).long().to(val_x.device)
        else:
            val_lbl_t = val_labels.clone().detach().long().to(val_x.device)

        # 使用模型进行预测
        predictions, logits = model(val_x)
        predictions = predictions.cpu().numpy()
        val_y = val_y.cpu().numpy()

        # 计算均方误差（MSE）作为验证损失
        mse_loss = F.mse_loss(torch.tensor(predictions).float().squeeze(), torch.tensor(val_y).float().squeeze())
        true_cls = val_lbl_t[:, -1].squeeze()  # [B]
        pred_cls = logits.argmax(dim=1)  # [B]
        acc = (pred_cls == true_cls).float().mean()  # 标量

        return mse_loss, acc


def plot_generator_losses(data_G, output_dir):
    """
    绘制 G1、G2、G3 的损失曲线。

    Args:
        data_G1 (list): G1 的损失数据列表，包含 [histD1_G1, histD2_G1, histD3_G1, histG1]。
        data_G2 (list): G2 的损失数据列表，包含 [histD1_G2, histD2_G2, histD3_G2, histG2]。
        data_G3 (list): G3 的损失数据列表，包含 [histD1_G3, histD2_G3, histD3_G3, histG3]。
    """

    plt.rcParams.update({'font.size': 12})
    all_data = data_G
    N = len(all_data)
    plt.figure(figsize=(6 * N, 5))

    for i, data in enumerate(all_data):
        plt.subplot(1, N, i + 1)
        for j, acc in enumerate(data):
            plt.plot(acc, label=f"G{i + 1} vs D{j + 1}" if j < N - 1 else f"G{i + 1} Combined", linewidth=2)

        plt.xlabel("Epoch", fontsize=14)
        plt.ylabel("Loss", fontsize=14)
        plt.title(f"G{i + 1} Loss over Epochs", fontsize=16)
        plt.legend()
        plt.grid(True)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "generator_losses.png"), dpi=500)
    plt.close()


def plot_discriminator_losses(data_D, output_dir):
    """
    绘制 D1、D2、D3 的损失曲线。

    Args:
        data_D1 (list): D1 的损失数据列表，包含 [histD1_D1, histD2_D1, histD3_D1, histD1]。
        data_D2 (list): D2 的损失数据列表，包含 [histD1_D2, histD2_D2, histD3_D2, histD2]。
        data_D3 (list): D3 的损失数据列表，包含 [histD1_D3, histD2_D3, histD3_D3, histD3]。
    """
    plt.rcParams.update({'font.size': 12})
    N = len(data_D)
    plt.figure(figsize=(6 * N, 5))

    for i, data in enumerate(data_D):
        plt.subplot(1, N, i + 1)
        for j, acc in enumerate(data):
            plt.plot(acc, label=f"D{i + 1} vs G{j + 1}" if j < len(data)-1 else f"D{i + 1} Combined", linewidth=2)

        plt.xlabel("Epoch", fontsize=14)
        plt.ylabel("Loss", fontsize=14)
        plt.title(f"D{i + 1} Loss over Epochs", fontsize=16)
        plt.legend()
        plt.grid(True)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "discriminator_losses.png"), dpi=500)
    plt.close()


def visualize_overall_loss(histG, histD, output_dir):
    """
    绘制 G1、G2、G3、D1、D2、D3 的损失曲线。

    Args:
        histG (list): 各生成器的损失数据列表。
        histD (list): 各判别器的损失数据列表。
    """
    plt.rcParams.update({'font.size': 12})
    N = len(histG)
    plt.figure(figsize=(5 * N, 4))

    for i, (g, d) in enumerate(zip(histG, histD)):
        plt.plot(g, label=f"G{i + 1} Loss", linewidth=2)
        plt.plot(d, label=f"D{i + 1} Loss", linewidth=2)

    plt.xlabel("Epoch", fontsize=14)
    plt.ylabel("Loss", fontsize=14)
    plt.title("Generator & Discriminator Loss", fontsize=16)
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "overall_losses.png"), dpi=500)
    plt.close()


def plot_mse_loss(hist_MSE_G, hist_val_loss, num_epochs,
                  output_dir):
    """
    绘制训练过程中和验证集上的MSE损失变化曲线

    参数：
    hist_MSE_G1, hist_MSE_G2, hist_MSE_G3 : 训练过程中各生成器的MSE损失
    hist_val_loss1, hist_val_loss2, hist_val_loss3 : 验证集上各生成器的MSE损失
    num_epochs : 训练的epoch数
    """
    plt.rcParams.update({'font.size': 12})
    N = len(hist_MSE_G)
    plt.figure(figsize=(5 * N, 4))

    for i, (MSE, val_loss) in enumerate(zip(hist_MSE_G, hist_val_loss)):
        plt.plot(range(num_epochs), MSE, label=f"Train MSE G{i + 1}", linewidth=2)
        plt.plot(range(num_epochs), val_loss, label=f"Val MSE G{i + 1}", linewidth=2, linestyle="--")

    plt.title("MSE Loss for Generators (Train vs Validation)", fontsize=16)
    plt.xlabel("Epoch", fontsize=14)
    plt.ylabel("MSE", fontsize=14)
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "mse_losses.png"), dpi=500)
    plt.close()

def inverse_transform(predictions, scaler):
    """ 使用y_scaler逆转换预测结果 """
    return scaler.inverse_transform(predictions)


def compute_metrics(true_values, predicted_values):
    """计算MSE, MAE, RMSE, MAPE"""
    mse = mean_squared_error(true_values, predicted_values)
    mae = mean_absolute_error(true_values, predicted_values)
    rmse = np.sqrt(mse)
    mape = np.mean(np.abs((true_values - predicted_values) / true_values)) * 100
    per_target_mse = np.mean((true_values - predicted_values) ** 2, axis=0)  # 新增
    return mse, mae, rmse, mape, per_target_mse


def plot_fitting_curve(true_values, predicted_values, output_dir, model_name):
    """绘制拟合曲线并保存结果"""
    plt.rcParams.update({'font.size': 12})
    plt.figure(figsize=(10, 6))
    plt.plot(true_values, label='True Values', linewidth=2)
    plt.plot(predicted_values, label='Predicted Values', linewidth=2, linestyle='--')
    plt.title(f'{model_name} Fitting Curve', fontsize=16)
    plt.xlabel('Time', fontsize=14)
    plt.ylabel('Value', fontsize=14)
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f'{output_dir}/{model_name}_fitting_curve.png', dpi=500)
    plt.close()


def save_metrics(metrics, output_dir, model_name):
    """保存MSE, MAE, RMSE, MAPE到文件"""
    with open(f'{output_dir}/{model_name}_metrics.txt', 'w') as f:
        f.write("MSE: {}\n".format(metrics[0]))
        f.write("MAE: {}\n".format(metrics[1]))
        f.write("RMSE: {}\n".format(metrics[2]))
        f.write("MAPE: {}\n".format(metrics[3]))


def evaluate_best_models(generators, best_model_state, train_xes, train_y, test_xes, test_y, y_scaler, output_dir):
    N = len(generators)

    # 加载模型并设为 eval
    for i in range(N):
        generators[i].load_state_dict(best_model_state[i])
        generators[i].eval()

    train_y_inv = inverse_transform(train_y, y_scaler)
    test_y_inv = inverse_transform(test_y, y_scaler)

    train_preds_inv = []
    test_preds_inv = []
    train_metrics_list = []
    test_metrics_list = []

    with torch.no_grad():
        for i in range(N):
            train_pred, train_cls = generators[i](train_xes[i])
            train_pred = train_pred.cpu().numpy()
            train_pred_inv = inverse_transform(train_pred, y_scaler)
            train_preds_inv.append(train_pred_inv)
            train_metrics = compute_metrics(train_y_inv, train_pred_inv)
            train_metrics_list.append(train_metrics)
            plot_fitting_curve(train_y_inv, train_pred_inv, output_dir, f'G{i+1}_Train')
            print(f"Train Metrics for G{i+1}: MSE={train_metrics[0]:.4f}, MAE={train_metrics[1]:.4f}, RMSE={train_metrics[2]:.4f}, MAPE={train_metrics[3]:.4f}")
            logging.info(f"Train Metrics for G{i+1}: MSE={train_metrics[0]:.4f}, MAE={train_metrics[1]:.4f}, RMSE={train_metrics[2]:.4f}, MAPE={train_metrics[3]:.4f}")

        for i in range(N):
            test_pred, test_cls = generators[i](test_xes[i])
            test_pred = test_pred.cpu().numpy()
            test_pred_inv = inverse_transform(test_pred, y_scaler)
            test_preds_inv.append(test_pred_inv)
            test_metrics = compute_metrics(test_y_inv, test_pred_inv)
            test_metrics_list.append(test_metrics)
            plot_fitting_curve(test_y_inv, test_pred_inv, output_dir, f'G{i+1}_Test')
            print(f"Test Metrics for G{i+1}: MSE={test_metrics[0]:.4f}, MAE={test_metrics[1]:.4f}, RMSE={test_metrics[2]:.4f}, MAPE={test_metrics[3]:.4f}")
            logging.info(f"Test Metrics for G{i+1}: MSE={test_metrics[0]:.4f}, MAE={test_metrics[1]:.4f}, RMSE={test_metrics[2]:.4f}, MAPE={test_metrics[3]:.4f}")

    # 构造返回结果
    result = {
        "train_mse":  [m[0] for m in train_metrics_list],
        "train_mae":  [m[1] for m in train_metrics_list],
        "train_rmse": [m[2] for m in train_metrics_list],
        "train_mape": [m[3] for m in train_metrics_list],
        "train_mse_per_target": [m[4] for m in train_metrics_list],

        "test_mse":  [m[0] for m in test_metrics_list],
        "test_mae":  [m[1] for m in test_metrics_list],
        "test_rmse": [m[2] for m in test_metrics_list],
        "test_mape": [m[3] for m in test_metrics_list],
        "test_mse_per_target": [m[4] for m in test_metrics_list],
    }

    return result


# 这个是针对 小牛姐的VWAP、TWAP 任务的评估函数。正常使用不用看这段
# hello小牛姐，你在看这段代码吗？:)
def evaluate_best_models_vwap(generators, best_model_state, train_xes, train_y, test_xes,
                              test_y, y_scaler, output_dir):
    N = len(generators)
    # 加载模型并设为 eval
    for i in range(N):
        generators[i].load_state_dict(best_model_state[i])
        generators[i].eval()

    train_y_inv = inverse_transform(train_y, y_scaler)
    test_y_inv = inverse_transform(test_y, y_scaler)

    train_preds_inv = []
    test_preds_inv = []
    train_metrics_list = []
    test_metrics_list = []

    all_test_pred_inv = []  # 用于存储所有模型的预测结果
    all_test_y_inv = []  # 用于存储所有真实标签

    with torch.no_grad():
        for i in range(N):
            train_pred, train_cls = generators[i](train_xes[i])
            train_pred = train_pred.cpu().numpy()
            train_pred_inv = inverse_transform(train_pred, y_scaler)
            train_preds_inv.append(train_pred_inv)
            train_metrics = compute_metrics(train_y_inv, train_pred_inv)
            train_metrics_list.append(train_metrics)
            plot_fitting_curve(train_y_inv, train_pred_inv, output_dir, f'G{i + 1}_Train')
            print(
                f"Train Metrics for G{i + 1}: MSE={train_metrics[0]:.4f}, MAE={train_metrics[1]:.4f}, RMSE={train_metrics[2]:.4f}, MAPE={train_metrics[3]:.4f}")
            logging.info(
                f"Train Metrics for G{i + 1}: MSE={train_metrics[0]:.4f}, MAE={train_metrics[1]:.4f}, RMSE={train_metrics[2]:.4f}, MAPE={train_metrics[3]:.4f}")

        for i in range(N):
            test_pred, test_cls = generators[i](test_xes[i])
            test_pred = test_pred.cpu().numpy()
            test_pred_inv = inverse_transform(test_pred, y_scaler)
            test_preds_inv.append(test_pred_inv)
            test_metrics = compute_metrics(test_y_inv, test_pred_inv)
            test_metrics_list.append(test_metrics)

            plot_fitting_curve(test_y_inv, test_pred_inv, output_dir, f'G{i + 1}_Test')
            print(
                f"Test Metrics for G{i + 1}: MSE={test_metrics[0]:.4f}, MAE={test_metrics[1]:.4f}, "
                f"RMSE={test_metrics[2]:.4f}, MAPE={test_metrics[3]:.4f}")
            logging.info(
                f"Test Metrics for G{i + 1}: MSE={test_metrics[0]:.4f}, MAE={test_metrics[1]:.4f}, "
                f"RMSE={test_metrics[2]:.4f}, MAPE={test_metrics[3]:.4f}")

            # 将当前模型的预测结果和真实标签添加到列表中
            all_test_pred_inv.append(test_pred_inv.flatten())  # 扁平化一维数组，确保逐行写入
            all_test_y_inv.append(test_y_inv.flatten())

            # 将所有模型的预测结果和真实标签写入到单独的 CSV 文件中
        test_predictions = {
            "true": np.concatenate(all_test_y_inv),  # 真实标签
            "prediction": np.concatenate(all_test_pred_inv)  # 预测值
        }

        test_predictions_df = pd.DataFrame(test_predictions)

        # 将测试预测结果保存为 CSV 文件
        test_results_file = r'E:\Coding_path\GCA_lite\out_put\multi\test_predictions.csv'
        test_predictions_df.to_csv(test_results_file, mode='a', header=not pd.io.common.file_exists(test_results_file),
                                   index=False)

    # 构造返回结果
    result = {
        "train_mse": [m[0] for m in train_metrics_list],
        "train_mae": [m[1] for m in train_metrics_list],
        "train_rmse": [m[2] for m in train_metrics_list],
        "train_mape": [m[3] for m in train_metrics_list],
        "train_mse_per_target": [m[4] for m in train_metrics_list],

        "test_mse": [m[0] for m in test_metrics_list],
        "test_mae": [m[1] for m in test_metrics_list],
        "test_rmse": [m[2] for m in test_metrics_list],
        "test_mape": [m[3] for m in test_metrics_list],
        "test_mse_per_target": [m[4] for m in test_metrics_list],

        # 将测试预测结果和真实标签添加到结果中
        "all_test_pred_inv": all_test_pred_inv,
        "all_test_y_inv": all_test_y_inv
    }

    return result

