
import argparse
from time_series_gca import GCA_time_series
import pandas as pd
import os
import models
from utils.logger import setup_experiment_logging


def parse_feature_ranges(feature_str):
    """将'6-10,15-17'这样的字符串转换为[6,7,8,9,10,15,16,17]这样的列表"""
    ranges = feature_str.split(',')
    result = []
    for r in ranges:
        if '-' in r:
            start, end = map(int, r.split('-'))
            result.extend(range(start, end + 1))
        else:
            result.append(int(r))
    return result


def run_experiments(args):
    # 创建保存结果的CSV文件
    results_file = os.path.join(args.output_dir, "gca_GT_NPDC_market.csv")
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
        print("Output directory created")

    gca = GCA_time_series(args, args.N_pairs, args.batch_size, args.num_epochs,
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
        # for target,feature in zip(target_columns,feature_columns):
        # 运行实验，获取结果
        # 处理特征组
        target_feature_columns = []
        for group in args.feature_groups:
            columns = parse_feature_ranges(group)
            target_feature_columns.append(columns)

        # 添加目标列到每个特征组
        for feature_group in target_feature_columns:
            feature_group.extend(args.target_columns[0])
        print("Using features:", target_feature_columns)

        gca.process_data(args.data_path,args.start_row, args.end_row, target, target_feature_columns)
        gca.init_dataloader()
        gca.init_model(args.num_classes)

        logger = setup_experiment_logging(args.output_dir, vars(args))

        if args.mode == "train":
            results = gca.train(logger)
        elif args.mode == "pred":
            results = gca.pred()

        # 将结果保存到CSV
        result_row = {
            "feature_columns": args.feature_groups,
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
            "test_mse_per_target": results["test_mse_per_target"]
        }
        df = pd.DataFrame([result_row])
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
    parser.add_argument('--notes', type=str, required=False, help="单个GAN的Transformer试试看",
                        default="")
    parser.add_argument('--data_path', type=str, required=False, help="输入数据的路径",
                        default="database/zx_processed_黄金_day.csv")
    parser.add_argument('--output_dir', type=str, required=False, help="输出文件的路径",
                        default="out_put/multi")
    parser.add_argument('--ckpt_dir', type=str, required=False, help="模型保存的路径",
                        default="ckpt")

    parser.add_argument('--feature_groups', nargs='+', type=str,
                        help="feature groups for each generator (e.g. '6-10,15-17' '3-5,11-14' '18-29')",
                        default=["1-4,6-10,15-17", "1-5,11-14", "1-4,18-29"])
    parser.add_argument('--target_columns', type=list, help="target to be predicted",
                        default=[list(range(1, 2))])


    parser.add_argument('--start_row', type=int, help="开始特征行", default=31)
    parser.add_argument('--end_row', type=int, help="结束特征行", default=-1)

    parser.add_argument('--window_sizes', nargs='+', type=int, help="滑动窗口大小", default=[15,15,15])
    parser.add_argument('--N_pairs', "-n", type=int, help="有几对生成器和判别器", default=3)
    parser.add_argument('--num_classes', "-n_cls", type=int, help="", default=3)
    parser.add_argument('--generators', "-gens", nargs='+', type=str, help="用哪一类生成器",
                        default=["transformer", "transformer", "transformer"])

    parser.add_argument('--discriminators', "-discs", type=list, help="names of discriminators", default=None)
    parser.add_argument('--distill_epochs', type=int, help="Epochs to do distillation", default=1)
    parser.add_argument('--cross_finetune_epochs', type=int, help="Epochs to do distillation", default=5)
    parser.add_argument('--device', type=list, help="用GPU", default=[0])

    parser.add_argument('--num_epochs', type=int, help="epoch", default=10000)
    parser.add_argument('--lr', type=int, help="initial learning rate", default=2e-5)
    parser.add_argument('--batch_size', type=int, help="Batch size for training", default=64)
    parser.add_argument('--train_split', type=float, help="Train-test split ratio", default=0.7)
    parser.add_argument('--random_seed', type=int, help="Random seed for reproducibility", default=3407)
    parser.add_argument("--amp_dtype", type=str, default="none",  choices=["float16", "bfloat16", "none"],
           help="自动混合精度类型（AMP）：float16, bfloat16, 或 none（禁用）")
    parser.add_argument('--mode', type=str, choices=["pred", "train"],
                        help="If train, it will also pred, while it predicts, it will laod the model checkpoint saved before.",
                        default="train")
    parser.add_argument("--ckpt_path", type=str, help="Checkpoint path", default="lastest")

    args = parser.parse_args()

    print("===== Running with the following arguments =====")
    for arg, value in vars(args).items():
        print(f"{arg}: {value}")
    print("===============================================")

    # 调用run_experiments函数
    run_experiments(args)
