"""
该代码实现了一个训练框架，用于训练一个生成模型（`generator`）。
在训练过程中，模型通过优化器 AdamW 进行优化，并使用 `ReduceLROnPlateau` 学习率调度器来调整学习率。
代码实现了一个基本的训练循环，其中包括损失函数的计算（分类损失和均方误差损失），模型的参数更新，
以及早停（Early Stopping）机制来防止过拟合。

主要功能：
1. **优化器与学习率调度器**：使用 `AdamW` 优化器对生成器进行优化，使用 `ReduceLROnPlateau` 来自动调整学习率。
2. **损失计算**：模型输出两个部分，分别计算分类损失（cross-entropy loss）和生成损失（均方误差损失 MSE）。两者加和得到总损失。
3. **早停机制**：引入 `patience_counter`，记录模型在验证集上的表现未改善的次数。如果验证损失在一定次数内没有提升，则提前停止训练。
4. **保存最佳模型**：在训练过程中，如果模型的验证损失比历史最好的损失更低，保存当前模型的状态，作为最佳模型。
5. **模型评估**：在训练结束后，使用验证集对最佳模型进行评估，输出训练结果。

核心逻辑：
1. 在每个 epoch 中，通过生成器模型计算预测值。
2. 计算分类损失和生成损失，优化模型参数。
3. 每个 epoch 后进行验证，计算验证损失，如果验证损失没有改进，则增加 `patience_counter`。
4. 如果 `patience_counter` 超过设定的耐心值 `patience`，则停止训练。
5. 训练结束后，返回最佳模型和训练结果。
"""


import copy
from .evaluate_visualization import *
import torch.optim.lr_scheduler as lr_scheduler
import torch.nn.functional as F


def train_baseframe(generator, dataloader,
                    y_scaler, train_x, train_y, val_x, val_y, val_label,
                    num_epochs,
                    output_dir,
                    device,
                    logger=None):
    g_learning_rate = 2e-5

    # 定义优化器 AdamW
    optimizers_G = torch.optim.AdamW(generator.parameters(), lr=g_learning_rate, betas=(0.9, 0.999))

    # 为每个优化器设置 ReduceLROnPlateau 调度器
    scheduler = lr_scheduler.ReduceLROnPlateau(optimizers_G, mode='min', factor=0.1, patience=16, min_lr=1e-7)
    best_epoch = -1

    # 定义生成历史记录的关键字
    """
    以三个为例，keys长得是这样得的：
    ['G1', 'G2', 'G3', 
    'D1', 'D2', 'D3', 
    'MSE_G1', 'MSE_G2', 'MSE_G3', 
    'val_G1', 'val_G2', 'val_G3', 
    'D1_G1', 'D2_G1', 'D3_G1', 'D1_G2', 'D2_G2', 'D3_G2', 'D1_G3', 'D2_G3', 'D3_G3'
    ]
    """
    keys = []
    g_keys = 'G1'
    MSE_g_keys = 'MSE_G1'
    val_loss_keys = 'val_G1'

    keys.extend(g_keys)
    keys.extend(MSE_g_keys)
    keys.extend(val_loss_keys)

    best_loss = 1000
    best_model_state = None

    """
    patience_counter 记录当前模型在验证集上的表现没有改善的次数。
    每当模型的验证损失（val_loss）没有变好时，patience_counter 就会增加 1。
    如果 val_loss 变得更小（即模型在验证集上表现得更好），patience_counter 就会被重置为 0，表示模型找到了更好的表现。
    如果 patience_counter 达到设定的最大容忍次数（patience），训练就会停止，避免过长时间训练导致过拟合。
    """
    patience_counter = 0
    patience = 50

    print("start training")
    for epoch in range(num_epochs):
        # epo_start = time.time()

        keys = []
        keys.extend(g_keys)
        keys.extend(MSE_g_keys)

        loss_dict = {key: [] for key in keys}

        for batch_idx, (x_last, y_last, label_last) in enumerate(dataloader):
            x_last = x_last.to(device)
            y_last = y_last.to(device)
            label_last = label_last.to(device)
            label_last = label_last.unsqueeze(-1)

            generator.train()
            outputs = generator(x_last)
            fake_data_G, fake_data_cls = outputs

            cls_loss = F.cross_entropy(fake_data_cls, label_last[:, -1, :].long().squeeze())
            mse_loss = F.mse_loss(fake_data_G.squeeze(), y_last[:, -1, :].squeeze())
            total_loss = cls_loss + mse_loss

            optimizers_G.zero_grad()
            total_loss.backward()
            optimizers_G.step()

            scheduler.step(total_loss)

        val_loss, acc = validate(generator, val_x, val_y, val_label)
        print(f'Validate MSE_loss: {val_loss}...')

        if val_loss > best_loss:
            patience_counter += 1
            print(f'patience last: {patience - patience_counter}, best: {best_loss}, val: {val_loss}')
        else:
            patience_counter = 0
            best_model_state = copy.deepcopy(generator.state_dict())
            best_loss = val_loss
        if patience_counter > patience:
            break
    results = evaluate_best_models_vwap([generator], [best_model_state], [train_x], train_y, [val_x], val_y, y_scaler,
                                   output_dir)
    return results, best_model_state
