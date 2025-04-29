"""
该模块提供了一个函数 `setup_experiment_logging()`，用于为实验过程设置日志记录。
日志文件将记录模型训练过程中的关键信息，方便调试和后续分析。

### 功能概述：
- **日志初始化**：`setup_experiment_logging()` 函数会创建并配置一个 `logging.Logger`，用来记录训练过程中产生的日志。日志文件会保存到指定的输出目录（`output_dir`）下，并且根据当前的时间戳为日志文件命名。
- **日志记录内容**：记录的信息包括训练过程中超参数设置、日志级别信息（如 INFO、ERROR 等），以及训练过程中产生的其他重要信息。
- **日志格式**：日志记录的格式包括时间戳、日志级别、日志内容，方便后期查看。

"""

import logging, json, os, datetime


def setup_experiment_logging(output_dir: str, args: dict,
                             log_name_prefix: str = "train") -> logging.Logger:
    """
    初始化实验日志。

    Parameters
    ----------
    output_dir : str
        训练脚本里传入的 output_dir，同 trainer 的可视化文件夹一致。
    args : dict
        所有需要记录的超参数 / CLI 解析结果，例如 vars(args)。
    log_name_prefix : str, optional
        生成文件名前缀，默认 "train"。

    Returns
    -------
    logging.Logger
        已经配置好的 logger，直接 logger.info(...) 使用。
    """
    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    log_dir = os.path.join(output_dir, "logs")
    os.makedirs(log_dir, exist_ok=True)
    log_path = os.path.join(log_dir, f"{log_name_prefix}_{ts}.log")

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(levelname)s  %(message)s",
        handlers=[
            logging.FileHandler(log_path, encoding="utf-8"),
            logging.StreamHandler()          # 保留控制台输出
        ]
    )

    logger = logging.getLogger()          # root logger
    logger.info("ARGS = %s",
                json.dumps(args, ensure_ascii=False, indent=2))
    return logger
