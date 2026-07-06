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
from cl_loss import get_cl_loss
from xgboost_module import get_feature, random_forest_classification, lightgbm_classification, \
    xgboost_classification, xgboost_regression_rmse, \
    random_forest_regression_rmse, lightgbm_regression_rmse, xgboost_regression_mae, random_forest_regression_mae, \
    lightgbm_regression_mae, select_best_model_and_predict

from utils import create_logger, seed_set, parse_args, print_result, show_tsne, \
    get_top_k_similar_feature, get_fingerprint, get_gate_weight_ratio, show_tsne_flag
from utils import NoamLR, build_scheduler, build_optimizer, get_metric_func
from utils import load_checkpoint, save_best_checkpoint, load_best_result

from loss import bulid_loss
from tqdm import tqdm

from dataset import build_loader
from dataset_enhancement import build_loader_enhancement

from model import build_model_MolME

from smiles_bert import SMILESTransformer
import matplotlib.pyplot as plt
import torch.nn as nn





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

                label_loss_i = criterion[i](y_pred, y_label)
                total_loss += label_loss_i

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
                y_label_np = y_label.cpu().numpy()
                if np.all(np.isnan(y_label_np)):
                    continue
                label_loss_i = criterion[i](y_pred, y_label)
                total_loss += label_loss_i
                y_pred = y_pred.detach().cpu().numpy()

            try:
                y_label_list[i].extend(y_label.cpu().numpy())
                y_pred_list[i].extend(y_pred)
            except:
                y_label_list[i] = []
                y_pred_list[i] = []
                y_label_list[i].extend(y_label.cpu().numpy())
                y_pred_list[i].extend(y_pred)

        if cfg.LOSS.CL_LOSS and vec1 is not None and vec2 is not None:
            ntx_loss = criterion[0].cl_loss(vec1, vec2)
            total_loss = total_loss + cfg.LOSS.ALPHA * ntx_loss

        if cfg.LOSS.CL_LOSS:
            cl_loss = get_cl_loss(graph_data, smiles_data)
            total_loss = total_loss + 0.08 * cl_loss

        optimizer.zero_grad()
        total_loss.backward()

        # 数值稳定性守卫：loss 或梯度非有限时跳过本次更新，避免 NaN/Inf 污染权重。
        # 对正常收敛的 run 永不触发，执行与结果保持完全一致。
        finite = bool(torch.isfinite(total_loss))
        if finite:
            for p in model.parameters():
                if p.grad is not None and not torch.isfinite(p.grad).all():
                    finite = False
                    break
        if finite:
            optimizer.step()
        else:
            logger.info(f'Warning: non-finite loss/grad at index {index}, skip optimizer step')

        if isinstance(lr_scheduler, NoamLR):
            lr_scheduler.step()

        losses.append(total_loss.item())

    # Compute metric
    results = []
    metric_func = get_metric_func(metric=cfg.DATA.METRIC)
    for i, task in enumerate(cfg.DATA.TASK_NAME):
        if i not in y_label_list or len(y_label_list[i]) == 0:
            continue
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

                label_loss_i = criterion[i](y_pred, y_label)
                total_loss += label_loss_i
                y_pred = F.softmax(y_pred.detach().cpu(), dim=-1)[:, 1].view(-1).numpy()
            else:
                y_pred = output[:, i]
                y_label = data.y[:, i]
                y_label_np = y_label.cpu().numpy()
                if np.all(np.isnan(y_label_np)):
                    continue
                label_loss_i = criterion[i](y_pred, y_label)
                total_loss += label_loss_i
                y_pred = y_pred.detach().cpu().numpy()

            try:
                y_label_list[i].extend(y_label.cpu().numpy())
                y_pred_list[i].extend(y_pred)
            except:
                y_label_list[i] = []
                y_pred_list[i] = []
                y_label_list[i].extend(y_label.cpu().numpy())
                y_pred_list[i].extend(y_pred)

        if cfg.LOSS.CL_LOSS and vec1 is not None and vec2 is not None:
            ntx_loss = criterion[0].cl_loss(vec1, vec2)
            total_loss = total_loss + cfg.LOSS.ALPHA * ntx_loss

        if cfg.LOSS.CL_LOSS:
            cl_loss = get_cl_loss(graph_data, smiles_data)
            total_loss = total_loss + 0.08 * cl_loss

        losses.append(total_loss.item())

    # Compute metric
    val_results = []
    metric_func = get_metric_func(metric=cfg.DATA.METRIC)
    for i, task in enumerate(cfg.DATA.TASK_NAME):
        if i not in y_label_list or len(y_label_list[i]) == 0:
            continue
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


def train(args, cfg, logger):
    seed_set(cfg.SEED)
    # step 1: dataloder loading, get number of tokens
    train_loader, val_loader, test_loader, weights = build_loader_enhancement(cfg, logger)

    # step 2: model loading
    model = build_model_MolME(cfg)

    logger.info(model)

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
    criterion = bulid_loss(cfg, weights)

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


    xgb_flag = True
    train_features, train_targets, train_smiles, train_gate_weights = get_feature(train_loader, model, xgb_flag, device)
    val_features, val_targets, val_smiles, val_gate_weights = get_feature(val_loader, model, xgb_flag, device)
    test_features, test_targets, test_smiles, test_gate_weights = get_feature(test_loader, model, xgb_flag, device)

    best_model_name, best_val_score, best_test_score = select_best_model_and_predict(
        cfg, train_features, train_targets, val_features, val_targets, test_features, test_targets, logger)

    # 8: Record training time.
    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))

    return test_score, best_test_score, best_val_score, best_model_name, total_time_str




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

    num_trials = 3
    print_result(cfg, args, logger, train, num_trials)





