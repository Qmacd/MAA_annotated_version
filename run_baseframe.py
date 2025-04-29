"""
===============================================================================
实验脚本: 运行模型实验

功能概述:
本脚本主要用于运行时间序列数据上的实验，使用 baseframe GAN 模型进行训练和预测。
实验流程包括数据处理、模型训练、预测、结果记录和模型保存。
通过命令行参数，用户可以灵活设置数据路径、实验参数、模型配置等，自动完成训练过程并将结果保存到指定的文件中。

主要功能:
1. **数据处理**: 通过指定数据路径，加载时间序列数据并进行预处理，包括特征列和目标列的提取、数据分割、归一化等。
2. **模型初始化**: 根据用户指定的生成器和判别器配置，初始化相应的模型架构。支持多种生成器（如 Transformer、LSTM）和判别器类型。
3. **训练模式**: 在训练模式下，进行模型训练，优化模型参数，并根据设定的超参数（如学习率、批量大小、训练周期数等）训练模型。
                在训练过程中，记录并保存训练过程中的详细日志。
4. **预测模式**: 在预测模式下，加载预训练模型并进行数据预测，输出预测结果。
5. **结果保存**: 将实验结果（如训练误差、测试误差等）保存到 CSV 文件中，便于后续分析和比较不同实验配置的效果。

配置和参数:
- 通过命令行解析器（`argparse`）提供丰富的实验参数，包括数据路径、特征列、目标列、学习率、训练周期数等。
- 支持自动混合精度训练（AMP），提高训练效率。
"""

import argparse
from time_series_baseframe import base_time_series
import pandas as pd
import os
import models
from utils.logger import setup_experiment_logging

# 主函数，用来运行实验。将实验参数传入，开始进行实验训练
def run_experiments(args):
    results_file = os.path.join(args.output_dir, "gca_GT_NPDC_market.csv")  # results_file是结果文件的路径，保存实验结果
    if not os.path.exists(args.output_dir):  # 如果不存在输出目录，则创建
        os.makedirs(args.output_dir)
        print("Output directory created")

    # 创建 base_time_series 类的实例 gca，并传入多个参数来配置实验。这些参数包括批量大小、训练的周期数、生成器和判别器的名称等。
    gca = base_time_series(args, args.N_pairs, args.batch_size, args.num_epochs,
                          args.generators, args.discriminators,
                          args.ckpt_dir, args.output_dir,
                          args.window_sizes,
                          ckpt_path=args.ckpt_path,
                          initial_learning_rate=args.lr,
                          train_split=args.train_split,
                          do_distill_epochs=args.distill_epochs,
                          cross_finetune_epochs=args.cross_finetune_epochs,
                          device=args.device,
                          seed=args.random_seed)

    for target in args.target_columns:
        """
        对于 args.target_columns 中的每个目标列（预测的目标变量），
        通过extend 将其添加到target_feature_columns 中，
        """
        target_feature_columns = args.feature_columns
        target_feature_columns.extend(target)
        print("using features:", target_feature_columns)

        gca.process_data(args.data_path, args.start_row, args.end_row, target, target_feature_columns)  # 处理数据，具体点进去看
        gca.init_dataloader()  # 初始化数据加载器，具体点进去看
        gca.init_model(args.num_classes)  # 初始化模型，具体点进去看

        logger = setup_experiment_logging(args.output_dir, vars(args))  # 设置实验日志记录器，用来记录实验过程中的详细信息。

        if args.mode == "train":
            results, best_model_state = gca.train(logger)  # 训练模型，并返回训练结果和最佳模型
            gca.save_models(best_model_state)  # 保存最佳模型
        elif args.mode == "pred":
            results = gca.pred()  # 模型预测并返回结果

        # 保存训练和测试结果到 CSV 文件中。每次实验结果会添加到结果文件 gca_GT_NPDC_market.csv 中。
        result_row = {
            "feature_columns": args.feature_columns,
            "target_columns": target,
            "train_mse": results["train_mse"],
            "train_mae": results["train_mae"],
            "train_rmse": results["train_rmse"],
            "train_mape": results["train_mape"],
            "train_mse_per_target": results["train_mse_per_target"],
            "test_mse": results["test_mse"],
            "test_mae": results["test_mae"],
            "test_rmse": results["test_rmse"],
            "test_mape": results["test_mape"],
            "test_mse_per_target": results["test_mse_per_target"],
        }
        df = pd.DataFrame([result_row])  # 构造一个 DataFrame，用来保存实验结果。
        df.to_csv(results_file, mode='a', header=not pd.io.common.file_exists(results_file), index=False)

if __name__ == "__main__":
    print("============= Available models ==================")
    for name in dir(models):
        obj = getattr(models, name)
        if isinstance(obj, type):
            print("\t", name)
    print("** Any other models please refer to add you model name to models.__init__ and import your costumed ones.")
    print("===============================================\n")

    # 使用argparse解析命令行参数
    parser = argparse.ArgumentParser(description="Run experiments for triple GAN model")
    parser.add_argument('--notes', type=str, required=False, help="做笔记的地方",
                        default="")
    parser.add_argument('--data_path', type=str, required=False, help="输入数据的路径",
                        default="database/VWAP/原油主连.csv")
    parser.add_argument('--output_dir', type=str, required=False, help="输出文件的路径",
                        default="out_put/multi")
    parser.add_argument('--ckpt_dir', type=str, required=False, help="模型保存的路径",
                        default="ckpt")

    parser.add_argument('--feature_columns', type=list, help="输入的特征列，",
                        default=list(range(1, 2)))
    parser.add_argument('--target_columns', type=list, help="预测的目标列",
                        default=[list(range(1, 2))])

    parser.add_argument('--start_row', type=int, help="开始特征行", default=31)
    parser.add_argument('--end_row', type=int, help="结束特征行", default=-1)

    parser.add_argument('--window_sizes', nargs='+', type=int, help="滑动窗口大小", default=[15])
    parser.add_argument('--N_pairs', "-n", type=int, help="有几对生成器和判别器", default=1)
    parser.add_argument('--num_classes', "-n_cls", type=int, help="这里是baseframe实验，不用管这个参数", default=3)
    parser.add_argument('--generators', "-gens", nargs='+', type=str, help="用哪一类生成器",
                        default=["transformer"])  # , "lstm", "transformer"
    parser.add_argument('--discriminators', "-discs", type=list, help="这里是baseframe实验，不用管这个参数",
                        default=None)
    parser.add_argument('--distill_epochs', type=int, help="这里是baseframe实验，不用管这个参数", default=0)
    parser.add_argument('--cross_finetune_epochs', type=int, help="这里是baseframe实验，不用管这个参数", default=0)
    parser.add_argument('--device', type=list, help="用GPU", default=[0])

    parser.add_argument('--num_epochs', type=int, help="epoch", default=10)
    parser.add_argument('--lr', type=int, help="initial learning rate", default=2e-5)
    parser.add_argument('--batch_size', type=int, help="Batch size for training", default=64)
    parser.add_argument('--train_split', type=float, help="训练集和测试集的比例", default=0.7)
    parser.add_argument('--random_seed', type=int, help="Random seed for reproducibility", default=3407)
    parser.add_argument("--amp_dtype",type=str, default="none", choices=["float16", "bfloat16", "none"],
                        help="自动混合精度类型（AMP）：float16, bfloat16, 或 none（禁用）")
    parser.add_argument('--mode', type=str, choices=["pred", "train"],
                        help="运行模式：pred表示预测，train表示训练",
                        default="train")
    parser.add_argument("--ckpt_path", type=str, help="指定权重路径", default="lastest")

    args = parser.parse_args()

    print("===== Running with the following arguments =====")
    for arg, value in vars(args).items():
        print(f"{arg}: {value}")
    print("===============================================")

    # 调用run_experiments函数
    run_experiments(args)
