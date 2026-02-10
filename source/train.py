# -*- coding: utf-8 -*-
"""
@Author  : Weimin Zhu
@Time    : 2021-10-01
@File    : train.py
"""

import os
import random
import time
import datetime
import argparse
import numpy as np

import torch
import torch.nn.functional as F
from sklearn.manifold import TSNE
from torch.utils.tensorboard import SummaryWriter
from transformers import RobertaConfig

from config import get_config
from cl_loss import get_total_loss
from xgboost_module import get_feature, random_forest_classification, lightgbm_classification, \
    xgboost_classification, xgboost_regression_rmse, \
    random_forest_regression_rmse, lightgbm_regression_rmse, xgboost_regression_mae, random_forest_regression_mae, \
    lightgbm_regression_mae

from utils import create_logger, seed_set, parse_args, print_result, show_tsne, \
    get_top_k_similar_feature, get_fingerprint, get_gate_weight_ratio, show_tsne_flag
from utils import NoamLR, build_scheduler, build_optimizer, get_metric_func
from utils import load_checkpoint, save_best_checkpoint, load_best_result

from loss import bulid_loss
from tqdm import tqdm

from dataset import build_loader
from data_util import build_loader_mergedataset
from dataset_enhancement import build_loader_enhancement

from model import build_model_HiGNN

from smiles_bert import SMILESTransformer
import matplotlib.pyplot as plt
import torch.nn as nn


# def parse_args():
#     parser = argparse.ArgumentParser(description="codes for HiGNN")
#
#     parser.add_argument(
#         "--cfg",
#         help="decide which cfg to use",
#         required=False,
#         default="../configs/merge_freeze.yaml",
#         type=str,
#     )
#
#     parser.add_argument(
#         "--opts",
#         help="Modify config options by adding 'KEY VALUE' pairs. ",
#         default=None,
#         nargs='+',
#     )
#
#     # easy config modification
#     parser.add_argument('--batch-size', type=int, help="batch size for training")
#     parser.add_argument('--lr_scheduler', type=str, help='learning rate scheduler')
#     parser.add_argument('--resume', help='resume from checkpoint')
#     parser.add_argument('--tag', help='tag of experiment')
#     parser.add_argument('--eval', action='store_true', help='Perform evaluation only')
#     parser.add_argument('--gpu', type=int, help='gpu',default=0)
#
#     args = parser.parse_args()
#     cfg = get_config(args)
#
#     return args, cfg


def train_one_epoch(cfg, model, criterion, trainloader, optimizer, lr_scheduler, device, logger, xgb_flag):
    model.train()

    losses = []
    y_pred_list = {}
    y_label_list = {}

    # i = 0
    for index, data  in enumerate(trainloader):
        # if data.get('smiles_vec') is None:
        #     get_smiles_vec(data)
        # print(index)
        data = data.to(device)
        # smiles_data
        # smiles_data = torch.squeeze(torch.tensor(data.smiles_vec), dim=1).cuda()
        # graph_data = model(data, xgb_flag)
        # graph_attended_by_seq = cross_attn_graph_to_seq_model(graph_data, smiles_data, smiles_data)
        # sequence_attended_by_graph = cross_attn_seq_to_graph_model(smiles_data, graph_data, graph_data)
        # combined_input = torch.stack([graph_attended_by_seq, sequence_attended_by_graph], dim=1)
        # output = multi_modal_transformer(combined_input)
        if cfg.LOSS.CL_LOSS:
            output, graph_data, smiles_data, vec1, vec2 = model(data, xgb_flag)
        else:
            output, graph_data, smiles_data, vec1, vec2 = model(data, xgb_flag)



        # 特征融合
        # output = None
        # output = fusion_model(output)
        # if isinstance(output, tuple):
        #     if cfg.LOSS.CL_LOSS:
        #         output, graph_data, smiles_data, vec1, vec2 = output
        #     else:
        #         output, graph_data, smiles_data = output
        # else:
        #     output, vec1, vec2 = output, None, None
        total_loss = 0
        label_loss = 0

        for i in range(len(cfg.DATA.TASK_NAME)):
            if cfg.DATA.TASK_TYPE == 'classification':
                y_pred = output[:, i * 2:(i + 1) * 2]
                y_label = data.y[:, i].squeeze()
                validId = np.where((y_label.cpu().numpy() == 0) | (y_label.cpu().numpy() == 1))[0]

                if len(validId) == 0:
                    continue
                if y_label.dim() == 0:
                    y_label = y_label.unsqueeze(0)

                y_pred = y_pred[torch.tensor(validId).to(device)]
                y_label = y_label[torch.tensor(validId).to(device)]

                label_loss += criterion[i](y_pred, y_label, vec1, vec2)
                local_total_loss, label_loss, cl_loss = get_total_loss(label_loss, graph_data, smiles_data, alpha=0.08)
                total_loss += local_total_loss
                if torch.isnan(y_pred).any():
                    print("y_pred contains NaN!")
                    print("index:{}".format(index))
                if torch.isinf(y_pred).any():
                    print("y_pred contains Inf!")
                    print("index:{}".format(index))

                y_pred = F.softmax(y_pred.detach().cpu(), dim=-1)[:, 1].view(-1).numpy()
            else:
                y_pred = output[:, i]
                y_label = data.y[:, i]
                label_loss += criterion(y_pred, y_label, vec1, vec2)
                local_total_loss, label_loss, cl_loss = get_total_loss(label_loss, graph_data, smiles_data, alpha=0.08)
                total_loss += local_total_loss
                y_pred = y_pred.detach().cpu().numpy()

            try:
                y_label_list[i].extend(y_label.cpu().numpy())
                y_pred_list[i].extend(y_pred)
            except:
                y_label_list[i] = []
                y_pred_list[i] = []
                y_label_list[i].extend(y_label.cpu().numpy())
                y_pred_list[i].extend(y_pred)

        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()

        if isinstance(lr_scheduler, NoamLR):
            lr_scheduler.step()

        losses.append(total_loss.item())

    # Compute metric
    results = []
    metric_func = get_metric_func(metric=cfg.DATA.METRIC)
    for i, task in enumerate(cfg.DATA.TASK_NAME):
        if cfg.DATA.TASK_TYPE == 'classification':
            nan = False
            if all(target == 0 for target in y_label_list[i]) or all(target == 1 for target in y_label_list[i]):
                nan = True
                logger.info(f'Warning: Found task "{task}" with targets all 0s or all 1s while training')

            if nan:
                results.append(float('nan'))
                continue

        if len(y_label_list[i]) == 0:
            continue

        # acc专用
        y_pred_list_copy = y_pred_list.copy()
        if  cfg.DATA.METRIC == 'acc':
            y_pred_list[i] = []
            threshold = 0.5
            for pred in y_pred_list_copy[i]:
                y_pred_list[i].append(1 if pred>=threshold else 0)
        results.append(metric_func(y_label_list[i], y_pred_list[i]))

    avg_results = np.nanmean(results)
    trn_loss = np.array(losses).mean()

    return trn_loss, avg_results


@torch.no_grad()
def validate(cfg, model, criterion, dataloader, epoch, device, logger, xgb_flag, eval_mode):
    model.eval()

    losses = []
    y_pred_list = {}
    y_label_list = {}

    for data in dataloader:
        # if data.get('smiles_vec') is None:
        #     get_smiles_vec(data)

        data = data.to(device)
        # smiles_data
        # smiles_data = torch.squeeze(torch.tensor(data.smiles_vec), dim=1).cuda()
        # graph_data = model(data, xgb_flag)
        # graph_attended_by_seq = cross_attn_graph_to_seq_model(graph_data, smiles_data, smiles_data)
        # sequence_attended_by_graph = cross_attn_seq_to_graph_model(smiles_data, graph_data, graph_data)
        # combined_input = torch.stack([graph_attended_by_seq, sequence_attended_by_graph], dim=1)
        # output = multi_modal_transformer(combined_input)

        if cfg.LOSS.CL_LOSS:
            output, graph_data, smiles_data, vec1, vec2 = model(data, xgb_flag)
        else:
            output, graph_data, smiles_data, vec1, vec2 = model(data, xgb_flag)

        # output, graph_data, smiles_data = model(data, xgb_flag)
        #
        # if isinstance(output, tuple):
        #     output, vec1, vec2 = output
        # else:
        #     output, vec1, vec2 = output, None, None
        total_loss = 0
        label_loss = 0

        for i in range(len(cfg.DATA.TASK_NAME)):
            if cfg.DATA.TASK_TYPE == 'classification':
                y_pred = output[:, i * 2:(i + 1) * 2]
                y_label = data.y[:, i].squeeze()
                # 筛选出有标签的数据，剔除掉无标签数据
                validId = np.where((y_label.cpu().numpy() == 0) | (y_label.cpu().numpy() == 1))[0]
                if len(validId) == 0:
                    continue
                if y_label.dim() == 0:
                    y_label = y_label.unsqueeze(0)

                y_pred = y_pred[torch.tensor(validId).to(device)]
                y_label = y_label[torch.tensor(validId).to(device)]

                label_loss += criterion[i](y_pred, y_label, vec1, vec2)
                local_total_loss, label_loss, cl_loss = get_total_loss(label_loss, graph_data, smiles_data, alpha=0.08)
                total_loss += local_total_loss
                y_pred = F.softmax(y_pred.detach().cpu(), dim=-1)[:, 1].view(-1).numpy()
            else:
                y_pred = output[:, i]
                y_label = data.y[:, i]
                label_loss += criterion(y_pred, y_label, vec1, vec2)
                local_total_loss, label_loss, cl_loss = get_total_loss(label_loss, graph_data, smiles_data, alpha=0.08)
                total_loss += local_total_loss
                y_pred = y_pred.detach().cpu().numpy()

            try:
                y_label_list[i].extend(y_label.cpu().numpy())
                y_pred_list[i].extend(y_pred)
            except:
                y_label_list[i] = []
                y_pred_list[i] = []
                y_label_list[i].extend(y_label.cpu().numpy())
                y_pred_list[i].extend(y_pred)
                losses.append(total_loss.item())

    # Compute metric
    val_results = []
    metric_func = get_metric_func(metric=cfg.DATA.METRIC)
    for i, task in enumerate(cfg.DATA.TASK_NAME):
        if cfg.DATA.TASK_TYPE == 'classification':
            nan = False
            if all(target == 0 for target in y_label_list[i]) or all(target == 1 for target in y_label_list[i]):
                nan = True
                logger.info(f'Warning: Found task "{task}" with targets all 0s or all 1s while validating')

            if nan:
                val_results.append(float('nan'))
                continue

        if len(y_label_list[i]) == 0:
            continue

        # acc专用
        y_pred_list_copy = y_pred_list.copy()
        if  cfg.DATA.METRIC == 'acc':
            y_pred_list[i] = []
            threshold = 0.5
            for pred in y_pred_list_copy[i]:
                y_pred_list[i].append(1 if pred>=threshold else 0)
        val_results.append(metric_func(y_label_list[i], y_pred_list[i]))

    avg_val_results = np.nanmean(val_results)
    val_loss = np.array(losses).mean()
    if eval_mode:
        logger.info(f'Seed {cfg.SEED} Dataset {cfg.DATA.DATASET} ==> '
                    f'The best epoch:{epoch} test_loss:{val_loss:.3f} test_scores:{avg_val_results:.3f}')
        return val_results

    return val_loss, avg_val_results




'''
利用预训练Bert模型(freeze)将所有SMILES转化为向量
'''
def get_smiles_vec(data):
    # 创建Bert模型
    bert_model = SMILESTransformer(bert_model_name='bert-base-uncased')
    # 获取SMILES_list
    smiles_list = data['smiles']
    # 获取SMILES_vec
    # print('*** train开始处理 ***')
    smiles_vec = bert_model(smiles_list)
    # 存入DataLoader
    data['smiles_vec'] = smiles_vec


# def show_tsne(train_features, val_features, test_features, train_targets, val_targets, test_targets):
#     # Step 1: 拼接所有特征和标签
#     all_features = torch.cat([train_features, val_features, test_features], dim=0)  # 拼接 Tensor
#     all_targets = torch.cat([train_targets, val_targets, test_targets], dim=0)      # 拼接标签
#
#     # 将拼接后的 Tensor 转换为 NumPy 数组
#     all_features = all_features.cpu().numpy()
#     all_targets = all_targets.cpu().numpy()
#
#     # 确保 all_targets 是一维数组
#     if all_targets.ndim > 1:
#         all_targets = np.squeeze(all_targets)
#
#     # Step 2: 使用 T-SNE 进行降维
#     tsne = TSNE(n_components=2, random_state=42, perplexity=30, learning_rate=200)
#     tsne_results = tsne.fit_transform(all_features)
#
#     # Step 3: 可视化 T-SNE 结果
#     fig, ax = plt.subplots(figsize=(8, 8))  # 创建 Figure 和 Axes 对象
#
#     # 根据 all_targets 的类别进行颜色区分
#     unique_classes = np.unique(all_targets)  # 获取唯一类别
#     colors = ['tomato', 'royalblue']  # 自定义颜色：红色表示 False（负类），蓝色表示 True（正类）
#
#     # 绘制散点图
#     for i, cls in enumerate(unique_classes):
#         indices = all_targets == cls
#         label = 'True' if cls == 1 else 'False'  # 根据类别值设置标签
#         ax.scatter(tsne_results[indices, 0], tsne_results[indices, 1],
#                    label=label, color=colors[i], alpha=0.6)
#
#     # 去掉刻度
#     ax.set_xticks([])
#     ax.set_yticks([])
#
#     # 去掉横轴和纵轴标签
#     ax.set_xlabel('')
#     ax.set_ylabel('')
#
#     # 添加图例
#     legend = ax.legend(
#         loc='upper right',                # 图例位置：右上角
#         fontsize=15,                      # 图例字体大小
#     )
#
#     # 调整布局以容纳图例，并减少外部空白
#     plt.tight_layout(rect=[0, 0, 0.85, 1])  # 留出右侧空间给图例
#     plt.subplots_adjust(left=0.05, right=0.95, top=0.95, bottom=0.05)  # 手动调整边距
#
#     # 显示图像
#     plt.show()



def train(args, cfg, logger):
    seed_set(cfg.SEED)
    # step 1: dataloder loading, get number of tokens
    ''' 原始DataLoader '''
    # train_loader, val_loader, test_loader, weights = build_loader(cfg, logger)
    ''' merge + DataLoader '''
    # train_loader, val_loader, test_loader, weights = build_loader_mergedataset(cfg, logger)
    ''' enhancement + DataLoader'''
    train_loader, val_loader, test_loader, weights = build_loader_enhancement(cfg, logger)

    # step 2: model loading
    # model = build_model(cfg)
    '''  Fine-tuning Bert  '''
    # model = build_model_ExtendedHiGNN(cfg)
    '''  (FreeBert + HiGNN)  OR  (Origin HiGNN)  '''
    model = build_model_HiGNN(cfg)

    logger.info(model)
    # SMILES vec
    # get_smiles_vec(cfg, train_loader, val_loader, test_loader)

    # print(" *** train_loader  smiles_vec ***")
    # for index, data in enumerate(tqdm(train_loader)):
    #     get_smiles_vec(data)
    # print(" *** val_loader  smiles_vec ***")
    # for index, data in enumerate(tqdm(val_loader)):
    #     get_smiles_vec(data)
    # print(" *** test_loader  smiles_vec ***")
    # for index, data in enumerate(tqdm(test_loader)):
    #     get_smiles_vec(data)

    # device mode
    device = torch.device("cuda:{}".format(args.gpu) if torch.cuda.is_available() else "cpu")
    model.to(device)


    # step 3: optimizer loading
    optimizer = build_optimizer(cfg, model)
    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"number of params: {n_parameters}")

    # step 4: lr_scheduler loading
    lr_scheduler = build_scheduler(cfg, optimizer, steps_per_epoch=len(train_loader))

    # step 5: loss function loading
    if weights is not None:
        criterion = [bulid_loss(cfg, torch.Tensor(w).to(device)) for w in weights]
    else:
        criterion = bulid_loss(cfg)

    # step 6: tensorboard loading
    if cfg.TRAIN.TENSORBOARD.ENABLE:
        tensorboard_dir = os.path.join(cfg.OUTPUT_DIR, "tensorboard")
        if not os.path.exists(tensorboard_dir):
            os.makedirs(tensorboard_dir)
    else:
        tensorboard_dir = None

    if tensorboard_dir is not None:
        writer = SummaryWriter(log_dir=tensorboard_dir)
    else:
        writer = None

    # step 7: model resuming (if training is interrupted, this will work.)
    best_epoch, best_score = 0, 0 if cfg.DATA.TASK_TYPE == 'classification' else float('inf')
    if cfg.TRAIN.RESUME:
        best_epoch, best_score = load_checkpoint(cfg, model, optimizer, lr_scheduler, logger)
        validate(cfg, model, criterion, val_loader, best_epoch, device, logger, False)

        if cfg.EVAL_MODE:
            return

    # step 8: training loop
    logger.info("Start training")
    early_stop_cnt = 0
    xgb_flag = False
    start_time = time.time()
    for epoch in range(cfg.TRAIN.START_EPOCH, cfg.TRAIN.MAX_EPOCHS):


        # 1: Results after one epoch training
        trn_loss, trn_score = train_one_epoch(cfg, model, criterion, train_loader, optimizer,
                                              lr_scheduler, device, logger, xgb_flag)
        val_loss, val_score = validate(cfg, model, criterion, val_loader, epoch, device, logger, xgb_flag, False)
        # Just for observing the testset results during training
        test_loss, test_score = validate(cfg, model, criterion, test_loader, epoch, device, logger, xgb_flag,  False)

        # 2: Upadate learning rate
        if not isinstance(lr_scheduler, NoamLR):
            lr_scheduler.step(val_loss)

        # 3: Print results
        if epoch % cfg.SHOW_FREQ == 0 or epoch == cfg.TRAIN.MAX_EPOCHS - 1:
            lr_cur = lr_scheduler.optimizer.param_groups[0]['lr']
            logger.info(f'Epoch:{epoch} {cfg.DATA.DATASET} trn_loss:{trn_loss:.3f} '
                        f'trn_{cfg.DATA.METRIC}:{trn_score:.3f} lr:{lr_cur:.5f}')
            logger.info(f'Epoch:{epoch} {cfg.DATA.DATASET} val_loss:{val_loss:.3f} '
                        f'val_{cfg.DATA.METRIC}:{val_score:.3f} lr:{lr_cur:.5f}')
            logger.info(f'Epoch:{epoch} {cfg.DATA.DATASET} test_loss:{test_loss:.3f} '
                        f'test_{cfg.DATA.METRIC}:{test_score:.3f} lr:{lr_cur:.5f}')

        # 4: Tensorboard for training visualization.
        loss_dict, acc_dict = {"train_loss": trn_loss}, {f"train_{cfg.DATA.METRIC}": trn_score}
        loss_dict["valid_loss"], acc_dict[f"valid_{cfg.DATA.METRIC}"] = val_loss, val_score

        if cfg.TRAIN.TENSORBOARD.ENABLE:
            writer.add_scalars(f"scalar/{cfg.DATA.METRIC}", acc_dict, epoch)
            writer.add_scalars("scalar/loss", loss_dict, epoch)

        # 5: Save best results.
        if cfg.DATA.TASK_TYPE == 'classification' and val_score > best_score or \
                cfg.DATA.TASK_TYPE == 'regression' and val_score < best_score:
            best_score, best_epoch = val_score, epoch
            save_best_checkpoint(cfg, epoch, model, best_score, best_epoch, optimizer, lr_scheduler, logger)
            early_stop_cnt = 0
        else:
            early_stop_cnt += 1
        # 6: Early stopping.
        if early_stop_cnt > cfg.TRAIN.EARLY_STOP > 0:
            logger.info('Early stop hitted!')
            break

    if cfg.TRAIN.TENSORBOARD.ENABLE:
        writer.close()


    # 7: Evaluation.
    model, best_epoch = load_best_result(cfg, model, logger)
    test_score = validate(cfg, model, criterion, test_loader, best_epoch, device, logger, xgb_flag, True)


    '''
        训练XGBoost
    '''
    xgb_flag = True
    train_features, train_targets, train_smiles, train_gate_weights = get_feature(train_loader, model, xgb_flag, device)
    val_features, val_targets, val_smiles, val_gate_weights = get_feature(val_loader, model, xgb_flag, device)
    test_features, test_targets, test_smiles, test_gate_weights = get_feature(test_loader, model, xgb_flag, device)

    # 统计每个任务的Gate_weight所在区间比率
    weight_ratio = get_gate_weight_ratio(train_gate_weights, val_gate_weights, test_gate_weights)
    print(weight_ratio)


    # Top K Similarity
    top_k = 6
    random_query_ids = random.sample(range(1000, 2000), 5)
    # random_query_ids = [777, 1820, 1905, 1660, 1960] #  ['*', 's', '^', 'o', 'D']  # 星形、正方形、三角形、圆形、菱形
    top_k_features_list = []
    # 循环 100 次
    for query_id in random_query_ids:
        # 调用 get_top_k_similar_feature 函数
        top_k_indices, top_k_features, top_k_smiles, query_smiles = get_top_k_similar_feature(
            train_features, val_features, test_features,
            train_smiles, val_smiles, test_smiles,
            query_id, top_k
        )

        # 调用 get_fingerprint 函数
        get_fingerprint(query_smiles, top_k_smiles, query_id)
        top_k_features_list.append(top_k_features)



    # t-SNE
    # 普通散点图
    show_tsne(cfg, train_features, val_features, test_features, train_targets, val_targets, test_targets)
    # Top-K 散点图
    show_tsne_flag(cfg, train_features, val_features, test_features, train_targets, val_targets, test_targets, top_k_features_list)


    if cfg.DATA.TASK_TYPE == 'classification':
        ''' XGBoost '''
        xgb_score = xgboost_classification(cfg, train_features, train_targets, val_features, val_targets, test_features,
                                           test_targets)
        ''' Random Forest '''
        rf_score = random_forest_classification(cfg, train_features, train_targets, val_features, val_targets,
                                                test_features, test_targets)
        ''' lightGBM '''
        gbm_score = lightgbm_classification(cfg, train_features, train_targets, val_features, val_targets,
                                            test_features, test_targets)
    else:
        if cfg.DATA.METRIC == 'rmse':
            xgb_score = xgboost_regression_rmse(cfg, train_features, train_targets, val_features, val_targets, test_features,
                                               test_targets)
            rf_score = random_forest_regression_rmse(cfg, train_features, train_targets, val_features, val_targets,
                                                    test_features, test_targets)
            gbm_score = lightgbm_regression_rmse(cfg, train_features, train_targets, val_features, val_targets,
                                                test_features, test_targets)
        elif cfg.DATA.METRIC == 'mae':
            xgb_score = xgboost_regression_mae(cfg, train_features, train_targets, val_features, val_targets, test_features,
                                               test_targets)
            rf_score = random_forest_regression_mae(cfg, train_features, train_targets, val_features, val_targets,
                                                    test_features, test_targets)
            gbm_score = lightgbm_regression_mae(cfg, train_features, train_targets, val_features, val_targets,
                                                test_features, test_targets)

    # xgb_score, best_model_or_models = xgboost_classification_with_grid_search(cfg, train_features, train_targets, val_features, val_targets, test_features, test_targets)

    # 8: Record training time.
    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))

    # logger.info('#' * 20)
    # logger.info(f'Training time {total_time_str}')
    # logger.info('Dataset:{}'.format(cfg.DATA.DATASET))
    # logger.info('Seed:{}'.format(cfg.SEED))
    # logger.info('Aug factor:{}'.format(cfg.DATA.AUG_FACTOR))
    # logger.info('origin score:{}'.format(np.mean(test_score)))
    # logger.info('xgb score:{}'.format(xgb_score))
    # logger.info('rf score:{}'.format(rf_score))
    # logger.info('gbm score:{}'.format(gbm_score))
    # logger.info('#' * 20)

    return test_score, xgb_score, rf_score, gbm_score, total_time_str




if __name__ == "__main__":
    args, cfg = parse_args()

    logger = create_logger(cfg)

    # print config
    logger.info(cfg.dump())
    # print device mode
    if torch.cuda.is_available():
        logger.info('GPU {}...'.format(args.gpu))
    else:
        logger.info('CPU mode...')

    # test_score, xgb_score, rf_score, gbm_score, train_time = train(args, cfg, logger)

    num_trials = 1
    print_result(cfg, args, logger, train, num_trials)





