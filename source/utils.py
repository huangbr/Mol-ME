# -*- coding: utf-8 -*-
"""
@Author  : Weimin Zhu
@Time    : 2021-09-28
@File    : utils.py
"""
import argparse
import os
import csv
import time
import math
import random
import logging
import numpy as np
import pandas as pd
from IPython.core.display_functions import display
from matplotlib import pyplot as plt
from rdkit.Chem import AllChem, rdmolops
from rdkit import Chem
from rdkit.Chem import Draw
from rdkit.Chem.Draw import MolsToGridImage
from rdkit.DataStructs import TanimotoSimilarity, DiceSimilarity
from sklearn.manifold import TSNE
from termcolor import colored

import torch
from torch.optim.lr_scheduler import _LRScheduler

from sklearn.metrics import auc, mean_absolute_error, mean_squared_error, precision_recall_curve, roc_auc_score, accuracy_score

from config import get_config
from IPython.display import SVG
import torch.nn.functional as F


# -----------------------------------------------------------------------------
# Set seed for random, numpy, torch, cuda.
# -----------------------------------------------------------------------------
def seed_set(seed=2021):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True


# -----------------------------------------------------------------------------
# Model resuming & checkpoint loading and saving.
# -----------------------------------------------------------------------------
def load_checkpoint(cfg, model, optimizer, lr_scheduler, logger):
    logger.info(f"==============> Resuming form {cfg.TRAIN.RESUME}....................")

    checkpoint = torch.load(cfg.TRAIN.RESUME, map_location='cpu')
    msg = model.load_state_dict(checkpoint['model'], strict=False)
    logger.info(msg)
    best_epoch, best_auc = 0, 0.0
    if not cfg.EVAL_MODE and 'optimizer' in checkpoint and 'lr_scheduler' in checkpoint and 'epoch' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer'])
        lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
        cfg.defrost()
        cfg.TRAIN.START_EPOCH = checkpoint['epoch'] + 1
        cfg.freeze()
        logger.info(f"=> loaded successfully '{cfg.TRAIN.RESUME}' (epoch {checkpoint['epoch']})")
        if 'best_auc' in checkpoint:
            best_auc = checkpoint['best_auc']
        if 'best_epoch' in checkpoint:
            best_epoch = checkpoint['best_epoch']

    del checkpoint
    torch.cuda.empty_cache()
    return best_epoch, best_auc


def save_best_checkpoint(cfg, epoch, model, best_auc, best_epoch, optimizer, lr_scheduler, logger):
    save_state = {'model': model.state_dict(),
                  'optimizer': optimizer.state_dict(),
                  'lr_scheduler': lr_scheduler.state_dict(),
                  'best_auc': best_auc,
                  'best_epoch': best_epoch,
                  'epoch': epoch,
                  'config': cfg}

    ckpt_dir = os.path.join(cfg.OUTPUT_DIR, "checkpoints")
    if not os.path.exists(ckpt_dir):
        os.makedirs(ckpt_dir)
    save_path = os.path.join(ckpt_dir, f'best_ckpt.pth')
    torch.save(save_state, save_path)
    logger.info(f"best_ckpt saved !!!")


def load_best_result(cfg, model, logger):
    ckpt_dir = os.path.join(cfg.OUTPUT_DIR, "checkpoints")
    best_ckpt_path = os.path.join(ckpt_dir, f'best_ckpt.pth')
    logger.info(f'Ckpt loading: {best_ckpt_path}')
    ckpt = torch.load(best_ckpt_path)
    model.load_state_dict(ckpt['model'])
    best_epoch = ckpt['best_epoch']

    return model, best_epoch


# -----------------------------------------------------------------------------
# Log
# -----------------------------------------------------------------------------
def create_logger(cfg):
    # log name
    dataset_name = cfg.DATA.DATASET
    tag_name = cfg.TAG
    time_str = time.strftime("%Y-%m-%d")
    log_name = "{}_{}_{}.log".format(dataset_name, tag_name, time_str)

    # log dir
    log_dir = os.path.join(cfg.OUTPUT_DIR, "logs")
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    # create logger
    logger = logging.getLogger(log_name)
    logger.setLevel(logging.DEBUG)
    logger.propagate = False

    # create formatter
    fmt = '[%(asctime)s] (%(filename)s %(lineno)d): %(levelname)s %(message)s'
    color_fmt = \
        colored('[%(asctime)s]', 'green') + \
        colored('(%(filename)s %(lineno)d): ', 'yellow') + \
        colored('%(levelname)-5s', 'magenta') + ' %(message)s'

    # create console handlers for master process
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.DEBUG)
    console_handler.setFormatter(
        logging.Formatter(fmt=color_fmt, datefmt='%Y-%m-%d %H:%M:%S'))
    logger.addHandler(console_handler)

    # create file handlers
    file_handler = logging.FileHandler(os.path.join(log_dir, log_name))
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(logging.Formatter(fmt=fmt, datefmt='%Y-%m-%d %H:%M:%S'))
    logger.addHandler(file_handler)

    return logger


# -----------------------------------------------------------------------------
# Data utils
# -----------------------------------------------------------------------------
def get_header(path):
    with open(path) as f:
        header = next(csv.reader(f))

    return header


def get_task_names(path, use_compound_names=False):
    index = 2 if use_compound_names else 1
    task_names = get_header(path)[index:]

    return task_names


# -----------------------------------------------------------------------------
# Optimizer
# -----------------------------------------------------------------------------
def build_optimizer(cfg, model):
    params = model.parameters()

    opt_lower = cfg.TRAIN.OPTIMIZER.TYPE.lower()
    optimizer = None

    if opt_lower == 'sgd':
        optimizer = torch.optim.SGD(
            params,
            lr=cfg.TRAIN.OPTIMIZER.BASE_LR,
            momentum=cfg.TRAIN.OPTIMIZER.MOMENTUM,
            weight_decay=cfg.TRAIN.OPTIMIZER.WEIGHT_DECAY,
            nesterov=True,
        )
    elif opt_lower == 'adam':
        optimizer = torch.optim.Adam(
            params,
            lr=cfg.TRAIN.OPTIMIZER.BASE_LR,
            weight_decay=cfg.TRAIN.OPTIMIZER.WEIGHT_DECAY,
        )
    return optimizer


# -----------------------------------------------------------------------------
# Lr_scheduler
# -----------------------------------------------------------------------------
def build_scheduler(cfg, optimizer, steps_per_epoch):
    if cfg.TRAIN.LR_SCHEDULER.TYPE == "reduce":
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode='min',
            factor=cfg.TRAIN.LR_SCHEDULER.FACTOR,
            patience=cfg.TRAIN.LR_SCHEDULER.PATIENCE,
            min_lr=cfg.TRAIN.LR_SCHEDULER.MIN_LR
        )
    elif cfg.TRAIN.LR_SCHEDULER.TYPE == "noam":
        scheduler = NoamLR(
            optimizer,
            warmup_epochs=[cfg.TRAIN.LR_SCHEDULER.WARMUP_EPOCHS],
            total_epochs=[cfg.TRAIN.MAX_EPOCHS],
            steps_per_epoch=steps_per_epoch,
            init_lr=[cfg.TRAIN.LR_SCHEDULER.INIT_LR],
            max_lr=[cfg.TRAIN.LR_SCHEDULER.MAX_LR],
            final_lr=[cfg.TRAIN.LR_SCHEDULER.FINAL_LR]
        )
    elif cfg.TRAIN.LR_SCHEDULER.TYPE == "cosine":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=5 * steps_per_epoch,  # 总迭代次数
            eta_min=1e-5              # 最小学习率
        )
    else:
        raise NotImplementedError("Unsupported LR Scheduler: {}".format(cfg.TRAIN.LR_SCHEDULER.TYPE))

    return scheduler


class NoamLR(_LRScheduler):
    def __init__(self, optimizer, warmup_epochs, total_epochs, steps_per_epoch,
                 init_lr, max_lr, final_lr):

        assert len(optimizer.param_groups) == len(warmup_epochs) == len(total_epochs) == len(init_lr) == \
               len(max_lr) == len(final_lr)

        self.num_lrs = len(optimizer.param_groups)

        self.optimizer = optimizer
        self.warmup_epochs = np.array(warmup_epochs)
        self.total_epochs = np.array(total_epochs)
        self.steps_per_epoch = steps_per_epoch
        self.init_lr = np.array(init_lr)
        self.max_lr = np.array(max_lr)
        self.final_lr = np.array(final_lr)

        self.current_step = 0
        self.lr = init_lr
        self.warmup_steps = (self.warmup_epochs * self.steps_per_epoch).astype(int)
        self.total_steps = self.total_epochs * self.steps_per_epoch
        self.linear_increment = (self.max_lr - self.init_lr) / self.warmup_steps

        self.exponential_gamma = (self.final_lr / self.max_lr) ** (1 / (self.total_steps - self.warmup_steps))

        super(NoamLR, self).__init__(optimizer)

    def get_lr(self):

        return list(self.lr)

    def step(self, current_step=None):

        if current_step is not None:
            self.current_step = current_step
        else:
            self.current_step += 1

        for i in range(self.num_lrs):
            if self.current_step <= self.warmup_steps[i]:
                self.lr[i] = self.init_lr[i] + self.current_step * self.linear_increment[i]
            elif self.current_step <= self.total_steps[i]:
                self.lr[i] = self.max_lr[i] * (self.exponential_gamma[i] ** (self.current_step - self.warmup_steps[i]))
            else:  # theoretically this case should never be reached since training should stop at total_steps
                self.lr[i] = self.final_lr[i]

            self.optimizer.param_groups[i]['lr'] = self.lr[i]


# -----------------------------------------------------------------------------
# Metric utils
# -----------------------------------------------------------------------------
def prc_auc(targets, preds):
    precision, recall, _ = precision_recall_curve(targets, preds)
    return auc(recall, precision)


def rmse(targets, preds):
    return math.sqrt(mean_squared_error(targets, preds))


def mse(targets, preds):
    return mean_squared_error(targets, preds)


def get_metric_func(metric):

    if metric == 'auc':
        return roc_auc_score

    if metric == 'prc':
        return prc_auc

    if metric == 'rmse':
        return rmse

    if metric == 'mae':
        return mean_absolute_error

    if metric == 'acc':
        return accuracy_score

    raise ValueError(f'Metric "{metric}" not supported.')



def parse_args():
    parser = argparse.ArgumentParser(description="codes for HiGNN")

    parser.add_argument(
        "--cfg",
        help="decide which cfg to use",
        required=False,
        default="../configs/merge_freeze.yaml",
        type=str,
    )

    parser.add_argument(
        "--opts",
        help="Modify config options by adding 'KEY VALUE' pairs. ",
        default=None,
        nargs='+',
    )

    # easy config modification
    parser.add_argument('--batch-size', type=int, help="batch size for training")
    parser.add_argument('--lr_scheduler', type=str, help='learning rate scheduler')
    parser.add_argument('--resume', help='resume from checkpoint')
    parser.add_argument('--tag', help='tag of experiment')
    parser.add_argument('--eval', action='store_true', help='Perform evaluation only')
    parser.add_argument('--gpu', type=int, help='gpu',default=0)

    args = parser.parse_args()
    cfg = get_config(args)

    return args, cfg



def print_result(cfg, args, logger, train, num_trials):
    test_score_list = []
    xgb_score_list = []
    rf_score_list = []
    gbm_score_list = []
    max_score_list = []
    min_score_list = []
    train_time_list = []

    if cfg.DATA.TASK_TYPE == 'classification':
        # Classification
        for trial in range(num_trials):
            test_scores, xgb_score, rf_score, gbm_score, train_time = train(args, cfg, logger)
            test_score = round(np.nanmean(test_scores)*100, 1)
            xgb_score = round(xgb_score*100, 1)
            rf_score = round(rf_score*100, 1)
            gbm_score = round(gbm_score*100, 1)

            test_score_list.append(test_score)
            xgb_score_list.append(xgb_score)
            rf_score_list.append(rf_score)
            gbm_score_list.append(gbm_score)
            train_time_list.append((train_time))
            max_score_list.append(max(test_score, xgb_score, rf_score, gbm_score))

        logger.info('*' * 40)
        logger.info('Dataset:{}'.format(cfg.DATA.DATASET))
        logger.info('Seed:{}'.format(cfg.SEED))
        logger.info('Aug factor:{}'.format(cfg.DATA.AUG_FACTOR))
        for trail in range(num_trials):
            logger.info('#' * 40)
            logger.info('Trial {}'.format(trail + 1))
            logger.info(f'Training time {train_time_list[trail]}')
            logger.info('origin score:{}'.format(test_score_list[trail]))
            logger.info('xgb score:{}'.format(xgb_score_list[trail]))
            logger.info('rf score:{}'.format(rf_score_list[trail]))
            logger.info('gbm score:{}'.format(gbm_score_list[trail]))
            logger.info('#' * 40)
        # 计算均值和标准差
        mean_accuracy = np.mean(max_score_list)
        std_accuracy = np.std(max_score_list)
        logger.info('Mean: {:.1f}'.format(float(mean_accuracy)))
        logger.info('Standard: {:.1f}'.format(float(std_accuracy)))
        logger.info('*' * 40)

    else:
        # Regression
        for trial in range(num_trials):
            test_scores, xgb_score, rf_score, gbm_score, train_time = train(args, cfg, logger)
            test_score = round(np.nanmean(test_scores), 6)
            xgb_score = round(xgb_score, 6)
            rf_score = round(rf_score, 6)
            gbm_score = round(gbm_score, 6)
            test_score_list.append(test_score)
            xgb_score_list.append(xgb_score)
            rf_score_list.append(rf_score)
            gbm_score_list.append(gbm_score)
            train_time_list.append((train_time))
            min_score_list.append(min(test_score, xgb_score, rf_score, gbm_score))

        logger.info('*' * 40)
        logger.info('Dataset:{}'.format(cfg.DATA.DATASET))
        logger.info('Seed:{}'.format(cfg.SEED))
        logger.info('Aug factor:{}'.format(cfg.DATA.AUG_FACTOR))
        if cfg.DATA.DATASET == 'qm8':
            for trail in range(num_trials):
                logger.info('#' * 40)
                logger.info('Trial {}'.format(trail + 1))
                logger.info(f'Training time {train_time_list[trail]}')
                logger.info('origin score:{:.5f}'.format(test_score_list[trail]))
                logger.info('xgb score:{:.5f}'.format(xgb_score_list[trail]))
                logger.info('rf score:{:.5f}'.format(rf_score_list[trail]))
                logger.info('gbm score:{:.5f}'.format(gbm_score_list[trail]))
                logger.info('#' * 40)
            # 计算均值和标准差
            mean_accuracy = np.mean(min_score_list)
            std_accuracy = np.std(min_score_list)
            logger.info('Mean: {:.5f}'.format(float(mean_accuracy)))
            logger.info('Standard: {:.5f}'.format(float(std_accuracy)))
            logger.info('*' * 40)

        elif cfg.DATA.DATASET == 'qm9':
            for trail in range(num_trials):
                logger.info('#' * 40)
                logger.info('Trial {}'.format(trail + 1))
                logger.info(f'Training time {train_time_list[trail]}')
                logger.info('origin score:{:.6f}'.format(test_score_list[trail]))
                logger.info('xgb score:{:.6f}'.format(xgb_score_list[trail]))
                logger.info('rf score:{:.6f}'.format(rf_score_list[trail]))
                logger.info('gbm score:{:.6f}'.format(gbm_score_list[trail]))
                logger.info('#' * 40)
            # 计算均值和标准差
            mean_accuracy = np.mean(min_score_list)
            std_accuracy = np.std(min_score_list)
            logger.info('Mean: {:.6f}'.format(float(mean_accuracy)))
            logger.info('Standard: {:.6f}'.format(float(std_accuracy)))
            logger.info('*' * 40)

        else:
            for trail in range(num_trials):
                logger.info('#' * 40)
                logger.info('Trial {}'.format(trail + 1))
                logger.info(f'Training time {train_time_list[trail]}')
                logger.info('origin score:{:.3f}'.format(test_score_list[trail]))
                logger.info('xgb score:{:.3f}'.format(xgb_score_list[trail]))
                logger.info('rf score:{:.3f}'.format(rf_score_list[trail]))
                logger.info('gbm score:{:.3f}'.format(gbm_score_list[trail]))
                logger.info('#' * 40)
            # 计算均值和标准差
            mean_accuracy = np.mean(min_score_list)
            std_accuracy = np.std(min_score_list)
            logger.info('Mean: {:.3f}'.format(float(mean_accuracy)))
            logger.info('Standard: {:.3f}'.format(float(std_accuracy)))
            logger.info('*' * 40)



# 普通散点图
def show_tsne(cfg, train_features, val_features, test_features, train_targets, val_targets, test_targets):

    # Step 1: 拼接所有特征和标签
    all_features = torch.cat([train_features, val_features, test_features], dim=0)  # 拼接 Tensor
    all_targets = torch.cat([train_targets, val_targets, test_targets], dim=0)      # 拼接标签
    all_features = all_features.cpu().numpy()
    all_targets = all_targets.cpu().numpy()
    # 确保 all_targets 是一维数组
    if all_targets.ndim > 1:
        all_targets = np.squeeze(all_targets)

    # Step 2: 使用 T-SNE 进行降维
    tsne = TSNE(n_components=2, random_state=42, perplexity=30, learning_rate=200)
    tsne_results = tsne.fit_transform(all_features)

    # Step 3: 根据任务类型绘制可视化图
    fig, ax = plt.subplots(figsize=(8, 8))  # 创建 Figure 和 Axes 对象

    if cfg.DATA.TASK_TYPE == 'classification':
        # 分类任务：使用类别区分颜色
        unique_classes = np.unique(all_targets)  # 获取唯一类别
        colors = ['tomato', 'royalblue']  # 自定义颜色：红色表示 False（负类），蓝色表示 True（正类）

        # 绘制散点图
        for i, cls in enumerate(unique_classes):
            indices = all_targets == cls
            label = 'True' if cls == 1 else 'False'  # 根据类别值设置标签
            ax.scatter(tsne_results[indices, 0], tsne_results[indices, 1],
                       label=label, color=colors[i], alpha=0.6, s=10)

        # 添加图例
        legend = ax.legend(
            loc='upper right',                # 图例位置：右上角
            fontsize=15,                      # 图例字体大小
        )



    elif cfg.DATA.TASK_TYPE == 'regression':
        # 回归任务：使用颜色映射表示目标值
        scatter = ax.scatter(
            tsne_results[:, 0], tsne_results[:, 1],
            c=all_targets,                    # 使用目标值作为颜色映射
            cmap='viridis',                   # 使用 viridis 颜色映射
            alpha=0.6,                        # 设置透明度
            s=10                              # 设置点的大小
        )

        # 添加颜色条，解释颜色与目标值的关系
        cbar = plt.colorbar(scatter)
        cbar.set_label("Hydration free energy (unit: kcal/mol)", fontsize=12)

    else:
        raise ValueError(f"Unsupported task type: {cfg.DATA.TASK_TYPE}")

    # 去掉刻度
    ax.set_xticks([])
    ax.set_yticks([])

    # 去掉横轴和纵轴标签
    ax.set_xlabel('')
    ax.set_ylabel('')

    # 调整布局以容纳图例或颜色条，并减少外部空白
    plt.tight_layout(rect=[0, 0, 0.85, 1])  # 留出右侧空间给图例或颜色条
    plt.subplots_adjust(left=0.05, right=0.95, top=0.95, bottom=0.05)  # 手动调整边距

    # 显示图像
    plt.show()






def show_tsne_flag(cfg, train_features, val_features, test_features, train_targets, val_targets, test_targets, top_k_features_list):
    # Step 1: 拼接所有特征和标签
    all_features = torch.cat([train_features, val_features, test_features], dim=0)  # 拼接 Tensor
    all_targets = torch.cat([train_targets, val_targets, test_targets], dim=0)      # 拼接标签
    all_features = all_features.cpu().numpy()
    all_targets = all_targets.cpu().numpy()

    # 确保 all_targets 是一维数组
    if all_targets.ndim > 1:
        all_targets = np.squeeze(all_targets)

    # Step 2: 使用 T-SNE 进行降维
    tsne = TSNE(n_components=2, random_state=42, perplexity=30, learning_rate=200)
    tsne_results = tsne.fit_transform(all_features)

    # Step 3: 根据任务类型绘制可视化图
    fig, ax = plt.subplots(figsize=(8, 8))  # 创建 Figure 和 Axes 对象

    if cfg.DATA.TASK_TYPE == 'classification':
        # 分类任务：使用类别区分颜色
        unique_classes = np.unique(all_targets)  # 获取唯一类别
        colors = ['tomato', 'royalblue']  # 自定义颜色：红色表示 False（负类），蓝色表示 True（正类）

        # 绘制所有特征的散点图
        for i, cls in enumerate(unique_classes):
            indices = all_targets == cls
            label = 'True' if cls == 1 else 'False'  # 根据类别值设置标签
            ax.scatter(tsne_results[indices, 0], tsne_results[indices, 1],
                       label=label, color=colors[i], alpha=0.6, s=30)

        # 添加图例
        legend = ax.legend(
            loc='upper right',                # 图例位置：右上角
            fontsize=15,                      # 图例字体大小
        )

        # 定义 Top-K 特征的颜色和标记形状
        top_k_markers = ['*', 's', '^', 'o', 'D']  # 星形、正方形、三角形、圆形、菱形

        # 循环处理每个 top_k_features
        for k, top_k_features in enumerate(top_k_features_list):
            # 打乱颜色顺序
            top_k_colors = ['green', 'yellow', 'purple', 'black', 'c']
            random.shuffle(top_k_colors)  # 随机打乱颜色顺序

            top_k_features = top_k_features.cpu().numpy()  # 转换为 NumPy 数组
            top_k_indices = [np.where((all_features == feature).all(axis=1))[0][0] for feature in top_k_features]
            top_k_tsne = tsne_results[top_k_indices]

            # 绘制 Top-K 特征
            for i, (x, y) in enumerate(top_k_tsne):
                ax.scatter(
                    x, y,
                    color=top_k_colors[i % len(top_k_colors)],  # 颜色循环
                    marker=top_k_markers[k],                   # 使用当前形状
                    s=300                                      # 散点大小
                )

    elif cfg.DATA.TASK_TYPE == 'regression':
        # 回归任务：使用颜色映射表示目标值
        scatter = ax.scatter(
            tsne_results[:, 0], tsne_results[:, 1],
            c=all_targets,                    # 使用目标值作为颜色映射
            cmap='viridis',                   # 使用 viridis 颜色映射
            alpha=0.6,                        # 设置透明度
            s=30                              # 增大散点大小到 50
        )

        # 添加颜色条，解释颜色与目标值的关系
        cbar = plt.colorbar(scatter)
        cbar.set_label("Hydration free energy (unit: kcal/mol)", fontsize=12)

        # 定义 Top-K 特征的颜色和标记形状
        top_k_markers = ['*', 's', '^', 'o', 'D']  # 星形、正方形、三角形、圆形、菱形

        # 循环处理每个 top_k_features
        for k, top_k_features in enumerate(top_k_features_list):
            # 打乱颜色顺序
            top_k_colors = ['green', 'yellow', 'purple', 'black', 'c']
            random.shuffle(top_k_colors)  # 随机打乱颜色顺序

            top_k_features = top_k_features.cpu().numpy()  # 转换为 NumPy 数组
            top_k_indices = [np.where((all_features == feature).all(axis=1))[0][0] for feature in top_k_features]
            top_k_tsne = tsne_results[top_k_indices]

            # 绘制 Top-K 特征
            for i, (x, y) in enumerate(top_k_tsne):
                ax.scatter(
                    x, y,
                    color=top_k_colors[i % len(top_k_colors)],  # 颜色循环
                    marker=top_k_markers[k],                   # 使用当前形状
                    s=300                                      # 散点大小
                )

    else:
        raise ValueError(f"Unsupported task type: {cfg.DATA.TASK_TYPE}")

    # 去掉刻度
    ax.set_xticks([])
    ax.set_yticks([])

    # 去掉横轴和纵轴标签
    ax.set_xlabel('')
    ax.set_ylabel('')

    # 调整布局以容纳图例或颜色条，并减少外部空白
    plt.tight_layout(rect=[0, 0, 0.85, 1])  # 留出右侧空间给图例或颜色条
    plt.subplots_adjust(left=0.05, right=0.95, top=0.95, bottom=0.05)  # 手动调整边距

    # 显示图像
    plt.show()





# def show_tsne(cfg, train_features, val_features, test_features, train_targets, val_targets, test_targets, top_k_features):
#     # Step 1: 拼接所有特征和标签
#     all_features = torch.cat([train_features, val_features, test_features], dim=0)  # 拼接 Tensor
#     all_targets = torch.cat([train_targets, val_targets, test_targets], dim=0)      # 拼接标签
#     all_features = all_features.cpu().numpy()
#     all_targets = all_targets.cpu().numpy()
#     top_k_features = top_k_features.cpu().numpy()
#
#     # 确保 all_targets 是一维数组
#     if all_targets.ndim > 1:
#         all_targets = np.squeeze(all_targets)
#
#     # Step 2: 使用 T-SNE 进行降维
#     tsne = TSNE(n_components=2, random_state=42, perplexity=30, learning_rate=200)
#     tsne_results = tsne.fit_transform(all_features)
#
#     # Step 3: 根据任务类型绘制可视化图
#     fig, ax = plt.subplots(figsize=(8, 8))  # 创建 Figure 和 Axes 对象
#
#     if cfg.DATA.TASK_TYPE == 'classification':
#         # 分类任务：使用类别区分颜色
#         unique_classes = np.unique(all_targets)  # 获取唯一类别
#         colors = ['tomato', 'royalblue']  # 自定义颜色：红色表示 False（负类），蓝色表示 True（正类）
#
#         # 绘制所有特征的散点图
#         for i, cls in enumerate(unique_classes):
#             indices = all_targets == cls
#             label = 'True' if cls == 1 else 'False'  # 根据类别值设置标签
#             ax.scatter(tsne_results[indices, 0], tsne_results[indices, 1],
#                        label=label, color=colors[i], alpha=0.6, s=30)  # 增大散点大小到 50
#
#         # 添加图例
#         legend = ax.legend(
#             loc='upper right',                # 图例位置：右上角
#             fontsize=15,                      # 图例字体大小
#         )
#
#         # 绘制 Top-K 特征
#         top_k_indices = [np.where((all_features == feature).all(axis=1))[0][0] for feature in top_k_features]
#         top_k_tsne = tsne_results[top_k_indices]
#
#         # 定义 Top-K 特征的颜色和标记形状
#         top_k_colors = ['green', 'yellow', 'purple', 'black', 'c']
#         top_k_markers = ['*', '*', '*', '*', '*']
#
#         # 绘制 Top-K 特征
#         for i in range(len(top_k_tsne)):
#             ax.scatter(top_k_tsne[i, 0], top_k_tsne[i, 1],
#                        color=top_k_colors[i], marker=top_k_markers[i], s=300)  # 增大散点大小到 200，并移除边框
# # *********************************************************************************************************************************
# # *********************************************************************************************************************************
#     elif cfg.DATA.TASK_TYPE == 'regression':
#         # 回归任务：使用颜色映射表示目标值
#         scatter = ax.scatter(
#             tsne_results[:, 0], tsne_results[:, 1],
#             c=all_targets,                    # 使用目标值作为颜色映射
#             cmap='viridis',                   # 使用 viridis 颜色映射
#             alpha=0.6,                        # 设置透明度
#             s=30                              # 增大散点大小到 50
#         )
#
#         # 添加颜色条，解释颜色与目标值的关系
#         cbar = plt.colorbar(scatter)
#         cbar.set_label("Hydration free energy (unit: kcal/mol)", fontsize=12)
#
#         # 绘制 Top-K 特征
#         top_k_indices = [np.where((all_features == feature).all(axis=1))[0][0] for feature in top_k_features]
#         top_k_tsne = tsne_results[top_k_indices]
#
#         # 定义 Top-K 特征的颜色和标记形状
#         top_k_colors = ['green', 'orange', 'purple', 'cyan', 'magenta']
#         top_k_markers = ['*', '*', '*', '*', '*']
#
#         # 绘制 Top-K 特征
#         for i in range(len(top_k_tsne)):
#             ax.scatter(top_k_tsne[i, 0], top_k_tsne[i, 1],
#                        color=top_k_colors[i], marker=top_k_markers[i], s=300)  # 增大散点大小到 200，并移除边框
#
#     else:
#         raise ValueError(f"Unsupported task type: {cfg.DATA.TASK_TYPE}")
#
#     # 去掉刻度
#     ax.set_xticks([])
#     ax.set_yticks([])
#
#     # 去掉横轴和纵轴标签
#     ax.set_xlabel('')
#     ax.set_ylabel('')
#
#     # 调整布局以容纳图例或颜色条，并减少外部空白
#     plt.tight_layout(rect=[0, 0, 0.85, 1])  # 留出右侧空间给图例或颜色条
#     plt.subplots_adjust(left=0.05, right=0.95, top=0.95, bottom=0.05)  # 手动调整边距
#
#     # 显示图像
#     plt.show()


# 1. 生成指纹
def generate_fingerprints(smiles_list, radius=3, nBits=2048):
    query_rdk_fps = []
    ecfp_fps = []
    for smiles in smiles_list:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            raise ValueError(f"Invalid SMILES: {smiles}")
        query_rdk_fp = rdmolops.RDKFingerprint(mol)
        ecfp_fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=nBits)
        query_rdk_fps.append(query_rdk_fp)
        ecfp_fps.append(ecfp_fp)
    return query_rdk_fps, ecfp_fps


# 2. 计算相似度
def calculate_similarity_tanimoto(query_fp, target_fps, similarity_metric=TanimotoSimilarity):
    similarities = [similarity_metric(query_fp, fp) for fp in target_fps]
    return similarities
def calculate_similarity_dice(query_fp, target_fps, similarity_metric=DiceSimilarity):
    similarities = [similarity_metric(query_fp, fp) for fp in target_fps]
    return similarities


# 3. 找到最相似的分子
def find_top_similar_molecules(query_smiles, target_smiles_list):
    query_mol = Chem.MolFromSmiles(query_smiles)
    # ECFP（Morgan）指纹
    query_ecfp_fp = AllChem.GetMorganFingerprintAsBitVect(query_mol, radius=3, nBits=2048)
    # 拓扑指纹
    query_rdk_fp = rdmolops.RDKFingerprint(query_mol)

    target_morgan_fps, target_ecfp_fps = generate_fingerprints(target_smiles_list)


    ecfp_similarities_tanimoto = calculate_similarity_tanimoto(query_ecfp_fp, target_ecfp_fps)
    rdk_fp_similarities_tanimoto = calculate_similarity_tanimoto(query_rdk_fp, target_morgan_fps)

    ecfp_similarities_dice = calculate_similarity_dice(query_ecfp_fp, target_ecfp_fps)
    rdk_fp_similarities_dice = calculate_similarity_dice(query_rdk_fp, target_morgan_fps)


    # average_similarities = [
    #     (rdk_fp_similarities[i] + ecfp_similarities[i]) / 2
    #     for i in range(len(rdk_fp_similarities))
    # ]
    data = {
        "Rank": range(1, len(target_smiles_list) + 1),
        "SMILES": target_smiles_list,
        "RDK_FP(tanimoto)": rdk_fp_similarities_tanimoto,
        "ECFP(tanimoto)": ecfp_similarities_tanimoto,
        "RDK_FP(dice)": rdk_fp_similarities_dice,
        "ECFP(dice)": ecfp_similarities_dice,
        # "AVG": average_similarities
    }
    result_df = pd.DataFrame(data)
    # 首先平均相似性分数降序排序
    # result_df = result_df.sort_values(by=["AVG"], ascending=False).reset_index(drop=True)
    result_df["Rank"] = result_df.index + 1
    return result_df


# 4. 可视化分子（适用于 PyCharm）
def visualize_molecules_pycharm(smiles_list, legends=None, molsPerRow=5, subImgSize=(200, 200)):
    # 将 SMILES 转换为 RDKit 分子对象
    mols = [Chem.MolFromSmiles(smiles) for smiles in smiles_list]

    # 检查分子对象是否成功生成
    if not all(mols):
        raise ValueError("部分 SMILES 字符串无法转换为分子对象，请检查输入。")

    # 使用 MolsToGridImage 生成分子网格图像
    try:
        img = Draw.MolsToGridImage(
            mols,
            molsPerRow=molsPerRow,
            subImgSize=subImgSize,
            legends=legends,
            useSVG=True
        )
    except AttributeError:
        # 如果 MolsToGridImage 不可用，使用替代方法
        img = Draw.MolsToImage(mols, subImgSize=subImgSize, legends=legends)

    # 显示图像
    if isinstance(img, SVG):  # 如果返回的是 SVG 对象
        return img
    else:  # 如果返回的是 PIL 图像对象
        img.show()


# 5. 可视化分子并保存到文件
def visualize_molecules_save_to_file(smiles_list, output_path="molecules.png", legends=None, molsPerRow=5,
                                     subImgSize=(200, 200)):
    mols = [Chem.MolFromSmiles(smiles) for smiles in smiles_list]
    img = Draw.MolsToGridImage(
        mols,
        molsPerRow=molsPerRow,
        subImgSize=subImgSize,
        legends=legends,
        useSVG=False  # 设置为 False 以生成 PNG 文件
    )
    # 保存图像到指定路径
    img.save(output_path)
    print(f"Image saved to {os.path.abspath(output_path)}")
    # 打开图像文件
    os.startfile(os.path.abspath(output_path))  # Windows 系统
    # 对于 macOS: os.system(f"open {output_path}")
    # 对于 Linux: os.system(f"xdg-open {output_path}")


def get_top_k_similar_feature(train_features, val_features, test_features, train_smiles, val_smiles, test_smiles, query_id, top_k):
    # 1. 拼接特征
    all_features = torch.cat([train_features, val_features, test_features], dim=0)

    # 2. 拼接 SMILES
    smiles_list = train_smiles + val_smiles + test_smiles

    # 3. 提取查询特征
    query_feature = all_features[query_id]
    query_smiles = smiles_list[query_id]

    # 4. 计算余弦相似度
    cosine_similarities = F.cosine_similarity(all_features, query_feature.unsqueeze(0), dim=1)
    # 防止出现 NaN 值（尽管 cosine_similarity 通常不会产生 NaN）
    cosine_similarities = torch.nan_to_num(cosine_similarities, nan=0.0)

    # 5.计算余弦距离
    cosine_distances = 1 - cosine_similarities

    # 6.排序并获取 Top-K 索引, 忽略余弦距离为 0 的结果
    top_k_indices = torch.argsort(cosine_distances)[:top_k].tolist()
    top_k_indices = [idx for idx in top_k_indices if cosine_distances[idx] > 0][:top_k]

    # 7.获取对应的 SMILES 字符串
    top_k_smiles = [smiles_list[idx] for idx in top_k_indices]
    top_k_features = all_features[top_k_indices]

    return top_k_indices, top_k_features, top_k_smiles, query_smiles





def get_fingerprint(query_smiles, top_k_smiles, query_id):

    # 计算ECFP、RDK_FP
    result_df = find_top_similar_molecules(query_smiles, top_k_smiles)
    legends = [f"Rank {i + 1}" for i in range(len(top_k_smiles))]

    print("***************************************************************************")
    print('query_id: {}'.format(query_id))
    print("query_smiles: {}".format(query_smiles))
    print("top_smiles: {}".format(top_k_smiles))
    print("legends: {}".format(legends))
    print("RDK_FP(tanimoto): {}".format(result_df["RDK_FP(tanimoto)"].tolist()))
    print("ECFP(tanimoto): {}".format(result_df["ECFP(tanimoto)"].tolist()))
    print("RDK_FP(dice): {}".format(result_df["RDK_FP(dice)"].tolist()))
    print("ECFP(dice): {}".format(result_df["ECFP(dice)"].tolist()))
    print("***************************************************************************")


    # 打印结果 DataFrame
    # print(result_df)
    # return result_df


def get_gate_weight_ratio(train_gate_weights, val_gate_weights, test_gate_weights):
    """
    统计 gate weights 在 [0, 25%), [25%, 50%), [50%, 75%), [75%, 100%] 区间的百分比。

    参数:
        train_gate_weights (torch.Tensor): 训练集的 gate weights。
        val_gate_weights (torch.Tensor): 验证集的 gate weights。
        test_gate_weights (torch.Tensor): 测试集的 gate weights。

    返回:
        dict: 各区间权重的百分比。
    """
    # Step 1: 将所有 gate weights 拼接成一个 Tensor
    all_gate_weights = torch.cat([train_gate_weights, val_gate_weights, test_gate_weights], dim=0)

    # Step 2: 确保数据在一维张量中
    all_gate_weights = all_gate_weights.view(-1)  # 展平为一维

    # Step 3: 定义区间边界
    bins = [0.0, 0.25, 0.5, 0.75, 1.0]

    # Step 4: 统计每个区间的权重数量
    hist = torch.histc(all_gate_weights, bins=len(bins) - 1, min=bins[0], max=bins[-1])

    # Step 5: 计算百分比
    total_count = hist.sum().item()
    percentages = (hist / total_count * 100).tolist()  # 转换为百分比形式

    # Step 6: 构造结果字典
    result = {
        "25p": percentages[0],
        "50p": percentages[1],
        "75p": percentages[2],
        "100p": percentages[3],
        "mean": all_gate_weights.mean().item()
    }

    return result
