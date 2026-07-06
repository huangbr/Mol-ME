import logging

import torch
from sklearn.metrics import roc_curve, auc, precision_recall_curve, roc_auc_score, mean_squared_error, \
    mean_absolute_error
from sklearn.model_selection import GridSearchCV
from xgboost import XGBClassifier, XGBRegressor
import pandas as pd
import numpy as np
import torch.nn.functional as F

from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
import lightgbm as lgb
def get_feature(data_loader, model, xgb_flag, device):
    total_features = []
    total_targets = []
    total_smiles =[]
    total_gate_weights = []
    model.eval()  # 确保模型处于评估模式
    with torch.no_grad():
        for index, data  in enumerate(data_loader):
            data = data.to(device)
            output, smiles_batch, gate_weight = model(data, xgb_flag)
            target = data.y
            # 将输出和目标移到 CPU 上以节省 GPU 显存
            output = output.cpu()
            target = target.cpu()

            # 收集特征和目标
            total_features.append(output)
            total_targets.append(target)
            total_smiles.append(smiles_batch)
            total_gate_weights.append(gate_weight)

            # 手动释放不再需要的张量
            del data, output, target
            torch.cuda.empty_cache()  # 清理缓存
    # 合并所有批次的结果
    total_features = torch.cat(total_features, dim=0)
    total_targets = torch.cat(total_targets, dim=0)
    total_gate_weights = torch.cat(total_gate_weights, dim=0)
    total_smiles = [smile for sublist in total_smiles for smile in sublist]

    return total_features, total_targets, total_smiles, total_gate_weights



def xgboost_classification(cfg, X_train, y_train, X_val, y_val, X_test, y_test):
    # 将所有输入张量移动到 CPU
    X_train = X_train.cpu()
    y_train = y_train.cpu()
    X_val = X_val.cpu()
    y_val = y_val.cpu()
    X_test = X_test.cpu()
    y_test = y_test.cpu()


    # 单标签分类的情况
    if y_test.shape[-1] == 1:
        # 将 y_train 和 y_test 转换为 NumPy 数组
        y_train = y_train.numpy().ravel()  # 展平为一维数组
        y_test = y_test.numpy().ravel()    # 展平为一维数组
        y_val = y_val.numpy().ravel()      # 展平为一维数组


        xgb_gbc = XGBClassifier(
            base_score=0.5, booster='gbtree', colsample_bylevel=1,
            colsample_bytree=1, gamma=1, learning_rate=0.01, max_delta_step=0,
            max_depth=4, min_child_weight=8, missing=np.nan, n_estimators=2000,
            n_jobs=1, nthread=None, objective='binary:logistic', random_state=0,
            reg_alpha=0, reg_lambda=1, scale_pos_weight=1, seed=None,
            silent=True, subsample=0.8, tree_method='hist',  # 使用 CPU 优化的树构建方法
            eval_metric='auc', early_stopping_rounds=300
        )


        xgb_gbc.fit(X_train.detach().numpy(), y_train, eval_set=[(X_val.detach().numpy(), y_val)])
        pre_pro = xgb_gbc.predict_proba(X_test.detach().numpy())[:, 1]

        fpr, tpr, threshold = roc_curve([float(i) for i in y_test], pre_pro)
        AUC = auc(fpr, tpr)
        # if args.folder == "benchmark_molnet/molnet_random_pcba_c" or args.folder == "benchmark_molnet/molnet_random_muv_c":
        #     precision, recall, _ = precision_recall_curve(y_test, pre_pro)
        #     AUC = auc(recall, precision)
        return AUC
    else:
        # 多标签分类的情况
        aucs = []

        # 如果 y_train 是三维的，将其转换为二维
        if len(y_train.shape) == 3:
            y_train = y_train.squeeze(1).cpu().numpy()  # 去掉第二维度并转换为 NumPy
            y_val = y_val.squeeze(1).cpu().numpy()      # 去掉第二维度并转换为 NumPy
            y_test = y_test.squeeze(1).cpu().numpy()    # 去掉第二维度并转换为 NumPy
        else:
            y_train = y_train.cpu().numpy()
            y_val = y_val.cpu().numpy()
            y_test = y_test.cpu().numpy()

        for i in range(y_test.shape[1]):
            # 遍历每一个多标签中的每一个标签
            if float(max(y_val[:, i])) == 0 or float(max(y_train[:, i])) == 0 or float(max(y_test[:, i])) == 0:
                continue

            # 过滤掉缺失的标签
            train_valid_indices = np.where((y_train[:,i] == 0) | (y_train[:,i] == 1))[0]
            valid_valid_indices = np.where((y_val[:,i] == 0) | (y_val[:,i] == 1))[0]
            test_valid_indices = np.where((y_test[:,i] == 0) | (y_test[:,i] == 1))[0]



            xgb_gbc = XGBClassifier(
                base_score=0.5, booster='gbtree', colsample_bylevel=1,
                colsample_bytree=1, gamma=1, learning_rate=0.1, max_delta_step=0,
                max_depth=4, min_child_weight=8, missing=np.nan, n_estimators=2000,
                n_jobs=1, nthread=None, objective='binary:logistic', random_state=0,
                reg_alpha=0, reg_lambda=1, scale_pos_weight=1, seed=None,
                silent=True, subsample=0.8, tree_method='hist',  # 使用 CPU 优化的树构建方法
                eval_metric='auc', early_stopping_rounds=300
            )
            xgb_gbc.fit(X_train[train_valid_indices].detach().numpy(), y_train[:, i][train_valid_indices], eval_set=[(X_val[valid_valid_indices].detach().numpy(), y_val[:, i][valid_valid_indices])])
            pre_pro = xgb_gbc.predict_proba(X_test[test_valid_indices].detach().numpy())
            y_pred = F.softmax(torch.tensor(pre_pro), dim=-1)[:, 1].view(-1).numpy()


            # valid_indices = np.where((y_test == 0) | (y_test == 1))[0]
            # y_test = y_test[valid_indices]
            # pre_pro = pre_pro[valid_indices]
            # fpr, tpr, threshold = roc_curve([float(j) for j in y_test[:, i]], pre_pro)
            # AUC = auc(fpr, tpr)
            AUC = roc_auc_score(y_test[:, i][test_valid_indices], y_pred)
            # if args.folder == "benchmark_molnet/molnet_random_pcba_c" or args.folder == "benchmark_molnet/molnet_random_muv_c":
            #     precision, recall, _ = precision_recall_curve([float(j) for j in y_test[:, i]], pre_pro)
            #     AUC = auc(recall, precision)
            aucs.append(AUC)
        return np.mean(aucs)



# 随机森林
def random_forest_classification(cfg, X_train, y_train, X_val, y_val, X_test, y_test):
    # 将所有输入张量移动到 CPU
    X_train = X_train.cpu()
    y_train = y_train.cpu()
    X_val = X_val.cpu()
    y_val = y_val.cpu()
    X_test = X_test.cpu()
    y_test = y_test.cpu()

    # 单标签分类的情况
    if y_test.shape[-1] == 1:
        # 转换为 NumPy 数组并展平
        y_train = y_train.numpy().ravel()
        y_test = y_test.numpy().ravel()
        y_val = y_val.numpy().ravel()

        logging.info('开始训练random forest!')
        rf_clf = RandomForestClassifier(
            n_estimators=200,
            max_depth=4,
            min_samples_split=8,
            n_jobs=-1,
            random_state=0
        )

        rf_clf.fit(X_train.detach().numpy(), y_train)
        pre_pro = rf_clf.predict_proba(X_test.detach().numpy())[:, 1]
        logging.info('训练结束random forest!')

        fpr, tpr, _ = roc_curve(y_test, pre_pro)
        AUC = auc(fpr, tpr)

        return AUC
    else:
        # 多标签分类的情况
        aucs = []

        # 如果 y_train 是三维的，将其转换为二维
        if len(y_train.shape) == 3:
            y_train = y_train.squeeze(1).cpu().numpy()
            y_val = y_val.squeeze(1).cpu().numpy()
            y_test = y_test.squeeze(1).cpu().numpy()
        else:
            y_train = y_train.cpu().numpy()
            y_val = y_val.cpu().numpy()
            y_test = y_test.cpu().numpy()

        for i in range(y_test.shape[1]):
            # 确保有正负样本才训练
            if float(max(y_val[:, i])) == 0 or float(max(y_train[:, i])) == 0 or float(max(y_test[:, i])) == 0:
                continue

            # 过滤掉缺失的标签
            train_valid_indices = np.where((y_train[:,i] == 0) | (y_train[:,i] == 1))[0]
            valid_valid_indices = np.where((y_val[:,i] == 0) | (y_val[:,i] == 1))[0]
            test_valid_indices = np.where((y_test[:,i] == 0) | (y_test[:,i] == 1))[0]

            rf_clf = RandomForestClassifier(
                n_estimators=200,
                max_depth=4,
                min_samples_split=8,
                n_jobs=-1,
                random_state=0
            )
            logging.info('开始训练random forest!')
            rf_clf.fit(X_train[train_valid_indices].detach().numpy(), y_train[:, i][train_valid_indices])
            pre_pro = rf_clf.predict_proba(X_test[test_valid_indices].detach().numpy())
            y_pred = F.softmax(torch.tensor(pre_pro), dim=-1)[:, 1].view(-1).numpy()
            logging.info('训练结束random forest!')

            AUC = roc_auc_score(y_test[:, i][test_valid_indices], y_pred)
            aucs.append(AUC)

        return np.mean(aucs)



def lightgbm_classification(cfg, X_train, y_train, X_val, y_val, X_test, y_test):

    # 将所有输入张量移动到 CPU 并转换为 NumPy 数组
    X_train = X_train.detach().cpu().numpy()
    y_train = y_train.detach().cpu().numpy()
    X_val = X_val.detach().cpu().numpy()
    y_val = y_val.detach().cpu().numpy()
    X_test = X_test.detach().cpu().numpy()
    y_test = y_test.detach().cpu().numpy()

    # 单标签分类的情况
    if y_test.shape[-1] == 1:
        # 转换为一维数组
        y_train = y_train.ravel()
        y_test = y_test.ravel()
        y_val = y_val.ravel()

        logging.info("Starting single-label classification training with LightGBM...")
        lgb_train = lgb.Dataset(X_train, y_train)
        lgb_eval = lgb.Dataset(X_val, y_val, reference=lgb_train)

        params = {
            'objective': 'binary',
            'metric': 'auc',
            'boosting_type': 'gbdt',
            'num_leaves': 31,
            'learning_rate': 0.05,
            'feature_fraction': 0.9,
            'bagging_fraction': 0.8,
            'bagging_freq': 5,
            'verbose': -1
        }

        # 使用回调函数设置早停机制
        callbacks = [lgb.early_stopping(stopping_rounds=300, verbose=False)]
        gbm = lgb.train(params,
                        lgb_train,
                        num_boost_round=2000,
                        valid_sets=[lgb_train, lgb_eval],
                        callbacks=callbacks)

        logging.info("Training completed.")
        pre_pro = gbm.predict(X_test, num_iteration=gbm.best_iteration)

        fpr, tpr, _ = roc_curve(y_test, pre_pro)
        AUC = auc(fpr, tpr)
        logging.info(f"Single-label AUC: {AUC}")

        return AUC
    else:
        # 多标签分类的情况
        aucs = []

        for i in range(y_test.shape[1]):
            logging.info(f"Training label {i+1}/{y_test.shape[1]}")

            # 确保有正负样本才训练
            if float(max(y_val[:, i])) == 0 or float(max(y_train[:, i])) == 0 or float(max(y_test[:, i])) == 0:
                logging.warning(f"No positive samples for label {i+1}, skipping.")
                continue

            # 过滤掉缺失的标签
            train_valid_indices = np.where((y_train[:, i] == 0) | (y_train[:, i] == 1))[0]
            valid_valid_indices = np.where((y_val[:, i] == 0) | (y_val[:, i] == 1))[0]
            test_valid_indices = np.where((y_test[:, i] == 0) | (y_test[:, i] == 1))[0]

            lgb_train = lgb.Dataset(X_train[train_valid_indices], y_train[:, i][train_valid_indices])
            lgb_eval = lgb.Dataset(X_val[valid_valid_indices], y_val[:, i][valid_valid_indices], reference=lgb_train)

            params = {
                'objective': 'binary',
                'metric': 'auc',
                'boosting_type': 'gbdt',
                'num_leaves': 31,
                'learning_rate': 0.05,
                'feature_fraction': 0.9,
                'bagging_fraction': 0.8,
                'bagging_freq': 5,
                'verbose': -1
            }

            # 使用回调函数设置早停机制
            callbacks = [lgb.early_stopping(stopping_rounds=300, verbose=False)]
            gbm = lgb.train(params,
                            lgb_train,
                            num_boost_round=2000,
                            valid_sets=[lgb_train, lgb_eval],
                            callbacks=callbacks)

            pre_pro = gbm.predict(X_test[test_valid_indices], num_iteration=gbm.best_iteration)

            AUC = roc_auc_score(y_test[:, i][test_valid_indices], pre_pro)
            aucs.append(AUC)

            logging.info(f"AUC for label {i+1}: {AUC}")

        mean_auc = np.mean(aucs)
        logging.info(f"Average AUC across all labels: {mean_auc}")
        return mean_auc





# *************************************************************************************************************************************
# *************************************************************************************************************************************
# *************************************************************************************************************************************




def xgboost_regression_rmse(cfg, X_train, y_train, X_val, y_val, X_test, y_test):

    # 将所有输入张量移动到 CPU 并转换为 NumPy 数组
    X_train = X_train.cpu().numpy()
    y_train = y_train.cpu().numpy().ravel()  # 展平为一维数组
    X_val = X_val.cpu().numpy()
    y_val = y_val.cpu().numpy().ravel()  # 展平为一维数组
    X_test = X_test.cpu().numpy()
    y_test = y_test.cpu().numpy().ravel()  # 展平为一维数组

    # 初始化 XGBoost 回归模型
    xgb_gbr = XGBRegressor(
        learn_rate=0.1,
        max_depth=4,  # 4
        min_child_weight=10,
        gamma=1,  # 1
        subsample=0.8,
        colsample_bytree=0.8,
        reg_alpha=0.8,
        objective='reg:linear',
        n_estimators=2000,
        tree_method='gpu_hist',
        n_gpus=-1,
        eval_metric='rmse',
        early_stopping_rounds=300
    )


    # 训练模型
    xgb_gbr.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)

    # 在测试集上进行预测
    y_pred = xgb_gbr.predict(X_test)


    mse = mean_squared_error(y_test, y_pred)  # 均方误差
    rmse = mse ** 0.5
    logging.info(f"xgb RMSE: {rmse:.4f}")
    return rmse


def random_forest_regression_rmse(cfg, X_train, y_train, X_val, y_val, X_test, y_test):
    X_train = X_train.cpu().numpy()
    X_val = X_val.cpu().numpy()
    X_test = X_test.cpu().numpy()

    # 确保 y_train, y_val, y_test 的形状正确
    if len(y_train.shape) == 1 or y_train.shape[1] == 1:
        y_train = y_train.ravel()
        y_val = y_val.ravel()
        y_test = y_test.ravel()

    # 单目标回归
    if len(y_test.shape) == 1 or y_test.shape[1] == 1:
        logging.info("Starting single-target regression training...")
        rf_reg = RandomForestRegressor(
            n_estimators=200,
            max_depth=8,
            min_samples_split=10,
            min_samples_leaf=4,
            max_features='log2',
            bootstrap=True,
            oob_score=False,  # 关闭，可能不影响模型，但节省计算
            random_state=42,
            n_jobs=-1
        )

        rf_reg.fit(X_train, y_train)
        y_pred = rf_reg.predict(X_test)
        mse = mean_squared_error(y_test, y_pred)
        rmse = mse ** 0.5
        logging.info(f"RF RMSE: {rmse:.4f}")
        return rmse

    # 多目标回归
    else:
        mses = []
        for i in range(y_test.shape[1]):
            logging.info(f"Training target {i + 1}/{y_test.shape[1]}")

            # 检查是否有有效数据
            if np.sum(~np.isnan(y_train[:, i])) == 0 or np.sum(~np.isnan(y_val[:, i])) == 0 or np.sum(~np.isnan(y_test[:, i])) == 0:
                logging.warning(f"No valid data for target {i + 1}, skipping.")
                continue

            # 过滤缺失值
            train_valid_indices = np.where(~np.isnan(y_train[:, i]))[0]
            test_valid_indices = np.where(~np.isnan(y_test[:, i]))[0]

            rf_reg = RandomForestRegressor(
                n_estimators=200,
                max_depth=8,
                min_samples_split=10,
                min_samples_leaf=4,
                max_features='log2',
                bootstrap=True,
                oob_score=False,  # 关闭，可能不影响模型，但节省计算
                random_state=42,
                n_jobs=-1
            )

            rf_reg.fit(X_train[train_valid_indices], y_train[:, i][train_valid_indices])
            y_pred = rf_reg.predict(X_test[test_valid_indices])
            mse = mean_squared_error(y_test[:, i][test_valid_indices], y_pred)
            mses.append(mse)

        mean_mse = np.mean(mses)
        rmse = mean_mse ** 0.5
        logging.info(f"RF MSE: {rmse:.4f}")
        return rmse


def lightgbm_regression_rmse(cfg, X_train, y_train, X_val, y_val, X_test, y_test):
    # 将数据从 GPU 移到 CPU 并转换为 NumPy 数组
    X_train = X_train.detach().cpu().numpy()
    y_train = y_train.detach().cpu().numpy()
    X_val = X_val.detach().cpu().numpy()
    y_val = y_val.detach().cpu().numpy()
    X_test = X_test.detach().cpu().numpy()
    y_test = y_test.detach().cpu().numpy()

    # 确保 y_train, y_val, y_test 的形状正确
    if len(y_train.shape) == 1 or y_train.shape[1] == 1:
        y_train = y_train.ravel()
        y_val = y_val.ravel()
        y_test = y_test.ravel()

    # 单目标回归
    if len(y_test.shape) == 1 or y_test.shape[1] == 1:
        logging.info("Starting single-target regression training with LightGBM...")
        lgb_train = lgb.Dataset(X_train, y_train)
        lgb_eval = lgb.Dataset(X_val, y_val, reference=lgb_train)

        params = {
            'objective': 'regression',
            'metric': 'rmse',
            'boosting_type': 'gbdt',
            'num_leaves': 63,  # 扩大叶子数量（2^6-1）
            'max_depth': 9,  # 加深树深度
            'learning_rate': 0.02,  # 降低学习率
            'feature_fraction': 0.9,  # 提高特征利用率
            'bagging_fraction': 0.7,  # 降低数据采样比例增强多样性
            'bagging_freq': 5,  # 每5次迭代执行bagging
            'min_data_in_leaf': 5,  # 减少叶节点最小数据量
            'lambda_l1': 1e-5,
            'lambda_l2': 0.1,  # 大幅降低L2正则化强度
            'min_gain_to_split': 0.1,  # 增加分裂增益阈值
            'random_state': 42
        }

        # 设置早停机制
        # callbacks = [lgb.early_stopping(stopping_rounds=150, verbose=False)]
        callbacks = [
            lgb.early_stopping(stopping_rounds=50),  # 早停窗口缩短
            lgb.log_evaluation(period=50)  # 每50轮输出日志
        ]
        gbm = lgb.train(params,
                        lgb_train,
                        num_boost_round=2000,
                        valid_sets=[lgb_train, lgb_eval],
                        callbacks=callbacks)

        logging.info("Training completed.")
        y_pred = gbm.predict(X_test, num_iteration=gbm.best_iteration)

        # 计算评估指标
        mse = mean_squared_error(y_test, y_pred)
        rmse = mse ** 0.5
        logging.info(f"LightGBM MSE: {rmse:.4f}")
        return rmse

    # 多目标回归
    else:
        mses = []
        for i in range(y_test.shape[1]):
            logging.info(f"Training target {i + 1}/{y_test.shape[1]}")

            # 检查是否有有效数据
            if np.sum(~np.isnan(y_train[:, i])) == 0 or np.sum(~np.isnan(y_val[:, i])) == 0 or np.sum(~np.isnan(y_test[:, i])) == 0:
                logging.warning(f"No valid data for target {i + 1}, skipping.")
                continue

            # 过滤缺失值
            train_valid_indices = np.where(~np.isnan(y_train[:, i]))[0]
            valid_valid_indices = np.where(~np.isnan(y_val[:, i]))[0]
            test_valid_indices = np.where(~np.isnan(y_test[:, i]))[0]

            lgb_train = lgb.Dataset(X_train[train_valid_indices], y_train[:, i][train_valid_indices])
            lgb_eval = lgb.Dataset(X_val[valid_valid_indices], y_val[:, i][valid_valid_indices], reference=lgb_train)


            params = {
                'objective': 'regression',
                'metric': 'rmse',  # 使用RMSE评估
                'boosting_type': 'gbdt',
                'num_leaves': 31,  # 与max_depth=6对应
                'max_depth': 6,  # 树最大深度
                'learning_rate': 0.05,  # 学习率
                'feature_fraction': 0.8,  # 特征采样比例
                'bagging_fraction': 0.8,  # 数据采样比例
                'bagging_freq': 1,  # 每次迭代执行bagging
                'min_data_in_leaf': 20,  # 防过拟合关键参数
                'lambda_l1': 1e-5,  # L1正则化
                'lambda_l2': 1,  # L2正则化
                'verbose': -1,  # 关闭日志
                'random_state': 42  # 随机种子
            }

            # 设置早停机制
            callbacks = [lgb.early_stopping(stopping_rounds=150, verbose=False)]
            gbm = lgb.train(params,
                            lgb_train,
                            num_boost_round=1000,
                            valid_sets=[lgb_train, lgb_eval],
                            callbacks=callbacks)

            y_pred = gbm.predict(X_test[test_valid_indices], num_iteration=gbm.best_iteration)

            # 计算评估指标
            mse = mean_squared_error(y_test[:, i][test_valid_indices], y_pred)
            logging.info(f"MSE for target {i + 1}: {mse:.4f}")
            mses.append(mse)

        mean_mse = np.mean(mses)
        rmse = mean_mse ** 0.5
        logging.info(f"LightGBM Mean MSE: {rmse:.4f}")
        return rmse





# **************************************************************************************************************************
# **************************************************************************************************************************
# **************************************************************************************************************************

# def xgboost_regression_mae(cfg, X_train, y_train, X_val, y_val, X_test, y_test):
#     # 将所有输入张量移动到 CPU 并转换为 NumPy 数组
#     X_train = X_train.cpu().numpy()
#     y_train = y_train.cpu().numpy().ravel()  # 展平为一维数组
#     X_val = X_val.cpu().numpy()
#     y_val = y_val.cpu().numpy().ravel()  # 展平为一维数组
#     X_test = X_test.cpu().numpy()
#     y_test = y_test.cpu().numpy().ravel()  # 展平为一维数组
#
#     # 初始化 XGBoost 回归模型
#     xgb_gbr = XGBRegressor(
#         objective= 'reg:squarederror',  # 使用MSE作为损失函数
#         eval_metric= 'mae',  # 使用MAE作为评价指标
#         learn_rate=0.1,
#         max_depth=4,  # 4
#         min_child_weight=10,
#         gamma=1,  # 1
#         subsample=0.8,
#         colsample_bytree=0.8,
#         reg_alpha=0.8,
#         n_estimators=2000,
#         tree_method='gpu_hist',
#         n_gpus=-1,
#         early_stopping_rounds=300
#     )
#
#     # 训练模型
#     xgb_gbr.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
#
#     # 在测试集上进行预测
#     y_pred = xgb_gbr.predict(X_test)
#
#     mae = mean_absolute_error(y_test, y_pred)  # 使用 MAE 作为评估指标
#     logging.info(f"xgb MAE: {mae:.4f}")
#     return mae

def xgboost_regression_mae(cfg, X_train, y_train, X_val, y_val, X_test, y_test):
    # 将所有输入张量移动到 CPU 并转换为 NumPy 数组
    X_train = X_train.cpu().numpy()
    y_train = y_train.cpu().numpy()
    X_val = X_val.cpu().numpy()
    y_val = y_val.cpu().numpy()
    X_test = X_test.cpu().numpy()
    y_test = y_test.cpu().numpy()

    # 单标签回归的情况
    if y_test.ndim == 1 or y_test.shape[1] == 1:
        # 如果 y_train 和 y_test 是二维但只有一列，将其展平为一维数组
        if y_train.ndim == 2:
            y_train = y_train.ravel()
        if y_test.ndim == 2:
            y_test = y_test.ravel()
        if y_val.ndim == 2:
            y_val = y_val.ravel()

        # 初始化 XGBoost 回归模型
        xgb_gbr = XGBRegressor(
            objective='reg:squarederror',  # 使用 MSE 作为损失函数
            eval_metric='mae',  # 使用 MAE 作为评价指标
            learning_rate=0.1,
            max_depth=4,  # 4
            min_child_weight=10,
            gamma=1,  # 1
            subsample=0.8,
            colsample_bytree=0.8,
            reg_alpha=0.8,
            n_estimators=2000,
            tree_method='hist',  # 使用 CPU 优化的树构建方法
            early_stopping_rounds=300
        )

        # 训练模型
        xgb_gbr.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)

        # 在测试集上进行预测
        y_pred = xgb_gbr.predict(X_test)

        # 计算 MAE
        mae = mean_absolute_error(y_test, y_pred)
        return mae

    else:
        # 多标签回归的情况
        maes = []

        # 遍历每一个目标标签
        for i in range(y_test.shape[1]):
            # 检查当前标签是否有效（非全零）
            if float(max(y_val[:, i])) == 0 and float(max(y_train[:, i])) == 0 and float(max(y_test[:, i])) == 0:
                continue

            # 过滤掉缺失的标签（假设标签值为 NaN 或其他无效值）
            train_valid_indices = np.where(np.isfinite(y_train[:, i]))[0]
            valid_valid_indices = np.where(np.isfinite(y_val[:, i]))[0]
            test_valid_indices = np.where(np.isfinite(y_test[:, i]))[0]

            # 初始化 XGBoost 回归模型
            xgb_gbr = XGBRegressor(
                objective='reg:squarederror',  # 使用 MSE 作为损失函数
                eval_metric='mae',  # 使用 MAE 作为评价指标
                learning_rate=0.1,
                max_depth=4,  # 4
                min_child_weight=10,
                gamma=1,  # 1
                subsample=0.8,
                colsample_bytree=0.8,
                reg_alpha=0.8,
                n_estimators=2000,
                tree_method='hist',  # 使用 CPU 优化的树构建方法
                early_stopping_rounds=300
            )

            # 训练模型
            xgb_gbr.fit(
                X_train[train_valid_indices],
                y_train[:, i][train_valid_indices],
                eval_set=[(X_val[valid_valid_indices], y_val[:, i][valid_valid_indices])],
                verbose=False
            )

            # 在测试集上进行预测
            y_pred = xgb_gbr.predict(X_test[test_valid_indices])

            # 计算 MAE
            mae = mean_absolute_error(y_test[:, i][test_valid_indices], y_pred)
            maes.append(mae)

        # 返回所有标签的平均 MAE
        return np.mean(maes)


def random_forest_regression_mae(cfg, X_train, y_train, X_val, y_val, X_test, y_test):
    # 将所有输入张量移动到 CPU 并转换为 NumPy 数组
    X_train = X_train.cpu().numpy()
    X_val = X_val.cpu().numpy()
    X_test = X_test.cpu().numpy()
    y_train = y_train.cpu().numpy()
    y_val = y_val.cpu().numpy()
    y_test = y_test.cpu().numpy()

    # 确保 y_train, y_val, y_test 的形状正确
    if len(y_train.shape) == 1:
        y_train = y_train.reshape(-1, 1)
        y_val = y_val.reshape(-1, 1)
        y_test = y_test.reshape(-1, 1)

    # 单目标回归
    if y_test.shape[1] == 1:
        logging.info("Starting single-target regression training...")
        rf_reg = RandomForestRegressor(
            criterion='squared_error',  # 使用 MSE 作为分裂标准
            n_estimators=200,
            max_depth=8,
            min_samples_split=10,
            min_samples_leaf=4,
            max_features='log2',
            bootstrap=True,
            oob_score=False,  # 关闭，可能不影响模型，但节省计算
            random_state=42,
            n_jobs=-1
        )

        rf_reg.fit(X_train, y_train.ravel())  # 展平为一维数组
        y_pred = rf_reg.predict(X_test)
        # 计算 MAE 作为评价指标
        mae = mean_absolute_error(y_test.ravel(), y_pred)
        print(f'MAE: {mae}')
        return mae

    # 多目标回归
    else:
        maes = []
        for i in range(y_test.shape[1]):
            logging.info(f"Training target {i + 1}/{y_test.shape[1]}")

            # 检查是否有有效数据
            if np.sum(~np.isnan(y_train[:, i])) == 0 or np.sum(~np.isnan(y_val[:, i])) == 0 or np.sum(
                    ~np.isnan(y_test[:, i])) == 0:
                logging.warning(f"No valid data for target {i + 1}, skipping.")
                continue

            # 过滤缺失值
            train_valid_indices = np.where(~np.isnan(y_train[:, i]))[0]
            test_valid_indices = np.where(~np.isnan(y_test[:, i]))[0]

            rf_reg = RandomForestRegressor(
                criterion='squared_error',  # 使用 MSE 作为分裂标准
                n_estimators=200,
                max_depth=8,
                min_samples_split=10,
                min_samples_leaf=4,
                max_features='log2',
                bootstrap=True,
                oob_score=False,  # 关闭，可能不影响模型，但节省计算
                random_state=42,
                n_jobs=-1
            )

            rf_reg.fit(X_train[train_valid_indices], y_train[:, i][train_valid_indices])
            y_pred = rf_reg.predict(X_test[test_valid_indices])
            mae = mean_absolute_error(y_test[:, i][test_valid_indices], y_pred)
            maes.append(mae)

        mean_mae = np.mean(maes)
        logging.info(f"RF MAE: {mean_mae:.4f}")
        return mean_mae

def lightgbm_regression_mae(cfg, X_train, y_train, X_val, y_val, X_test, y_test):
    # 将数据从 GPU 移到 CPU 并转换为 NumPy 数组
    X_train = X_train.detach().cpu().numpy()
    y_train = y_train.detach().cpu().numpy()
    X_val = X_val.detach().cpu().numpy()
    y_val = y_val.detach().cpu().numpy()
    X_test = X_test.detach().cpu().numpy()
    y_test = y_test.detach().cpu().numpy()

    # 确保 y_train, y_val, y_test 的形状正确
    if len(y_train.shape) == 1 or y_train.shape[1] == 1:
        y_train = y_train.ravel()
        y_val = y_val.ravel()
        y_test = y_test.ravel()

    # 单目标回归
    if len(y_test.shape) == 1 or y_test.shape[1] == 1:
        logging.info("Starting single-target regression training with LightGBM...")
        lgb_train = lgb.Dataset(X_train, y_train)
        lgb_eval = lgb.Dataset(X_val, y_val, reference=lgb_train)

        params = {
            'objective': 'regression',  # 使用 MSE 作为损失函数（默认回归任务）
            'metric': 'mae',  # 使用 MAE 作为评价指标
            'boosting_type': 'gbdt',
            'num_leaves': 63,  # 扩大叶子数量（2^6-1）
            'max_depth': 9,  # 加深树深度
            'learning_rate': 0.02,  # 降低学习率
            'feature_fraction': 0.9,  # 提高特征利用率
            'bagging_fraction': 0.7,  # 降低数据采样比例增强多样性
            'bagging_freq': 5,  # 每5次迭代执行bagging
            'min_data_in_leaf': 5,  # 减少叶节点最小数据量
            'lambda_l1': 1e-5,
            'lambda_l2': 0.1,  # 大幅降低L2正则化强度
            'min_gain_to_split': 0.1,  # 增加分裂增益阈值
            'verbose': -1,  # 彻底关闭日志
            'random_state': 42
        }

        # 设置早停机制
        # callbacks = [lgb.early_stopping(stopping_rounds=150, verbose=False)]
        callbacks = [
            lgb.early_stopping(stopping_rounds=50),  # 早停窗口缩短
        ]
        gbm = lgb.train(params,
                        lgb_train,
                        num_boost_round=2000,
                        valid_sets=[lgb_train, lgb_eval],
                        callbacks=callbacks)

        logging.info("Training completed.")
        y_pred = gbm.predict(X_test, num_iteration=gbm.best_iteration)

        # 计算评估指标
        mae = mean_absolute_error(y_test, y_pred)  # 测试集上的 MAE
        logging.info(f"LightGBM MAE: {mae:.4f}")
        return mae

    # 多目标回归
    else:
        maes = []
        for i in range(y_test.shape[1]):
            logging.info(f"Training target {i + 1}/{y_test.shape[1]}")

            # 检查是否有有效数据
            if np.sum(~np.isnan(y_train[:, i])) == 0 or np.sum(~np.isnan(y_val[:, i])) == 0 or np.sum(
                    ~np.isnan(y_test[:, i])) == 0:
                logging.warning(f"No valid data for target {i + 1}, skipping.")
                continue

            # 过滤缺失值
            train_valid_indices = np.where(~np.isnan(y_train[:, i]))[0]
            valid_valid_indices = np.where(~np.isnan(y_val[:, i]))[0]
            test_valid_indices = np.where(~np.isnan(y_test[:, i]))[0]

            lgb_train = lgb.Dataset(X_train[train_valid_indices], y_train[:, i][train_valid_indices])
            lgb_eval = lgb.Dataset(X_val[valid_valid_indices], y_val[:, i][valid_valid_indices], reference=lgb_train)

            params = {
                'objective': 'regression',  # 使用 MSE 作为损失函数（默认回归任务）
                'metric': 'mae',  # 使用 MAE 作为评价指标
                'boosting_type': 'gbdt',
                'num_leaves': 31,  # 与max_depth=6对应
                'max_depth': 6,  # 树最大深度
                'learning_rate': 0.05,  # 学习率
                'feature_fraction': 0.8,  # 特征采样比例
                'bagging_fraction': 0.8,  # 数据采样比例
                'bagging_freq': 1,  # 每次迭代执行bagging
                'min_data_in_leaf': 20,  # 防过拟合关键参数
                'lambda_l1': 1e-5,  # L1正则化
                'lambda_l2': 1,  # L2正则化
                'verbose': -1,  # 关闭日志
                'random_state': 42  # 随机种子
            }

            # 设置早停机制
            callbacks = [lgb.early_stopping(stopping_rounds=150, verbose=False)]
            gbm = lgb.train(params,
                            lgb_train,
                            num_boost_round=1000,
                            valid_sets=[lgb_train, lgb_eval],
                            callbacks=callbacks)

            y_pred = gbm.predict(X_test[test_valid_indices], num_iteration=gbm.best_iteration)

            # 计算评估指标
            mae = mean_absolute_error(y_test[:, i][test_valid_indices], y_pred)  # 测试集上的 MAE
            logging.info(f"MAE for target {i + 1}: {mae:.4f}")
            maes.append(mae)

        mean_mae = np.mean(maes)
        logging.info(f"LightGBM Mean MAE: {mean_mae:.4f}")
        return mean_mae


# ======================================================================================================================
# 新增函数：基于验证集选择最优模型 (对应论文 Equation 10)
# ======================================================================================================================

def train_and_evaluate_xgb_classification(cfg, X_train, y_train, X_val, y_val, X_test, y_test):
    """
    训练XGBoost分类模型，返回验证集分数和测试集预测
    """
    X_train_np = X_train.cpu().numpy()
    y_train_np = y_train.cpu().numpy()
    X_val_np = X_val.cpu().numpy()
    y_val_np = y_val.cpu().numpy()
    X_test_np = X_test.cpu().numpy()
    y_test_np = y_test.cpu().numpy()

    if y_test_np.shape[-1] == 1:
        y_train_np = y_train_np.ravel()
        y_val_np = y_val_np.ravel()
        y_test_np = y_test_np.ravel()

        model = XGBClassifier(
            base_score=0.5, booster='gbtree', colsample_bylevel=1,
            colsample_bytree=1, gamma=1, learning_rate=0.01, max_delta_step=0,
            max_depth=4, min_child_weight=8, missing=np.nan, n_estimators=2000,
            n_jobs=1, nthread=None, objective='binary:logistic', random_state=0,
            reg_alpha=0, reg_lambda=1, scale_pos_weight=1, seed=None,
            silent=True, subsample=0.8, tree_method='hist',
            eval_metric='auc', early_stopping_rounds=300
        )

        model.fit(X_train_np, y_train_np, eval_set=[(X_val_np, y_val_np)])

        val_pred = model.predict_proba(X_val_np)[:, 1]
        val_auc = roc_auc_score(y_val_np, val_pred)

        test_pred = model.predict_proba(X_test_np)[:, 1]
        test_auc = roc_auc_score(y_test_np, test_pred)

        return val_auc, test_auc, model
    else:
        if len(y_train_np.shape) == 3:
            y_train_np = y_train_np.squeeze(1)
            y_val_np = y_val_np.squeeze(1)
            y_test_np = y_test_np.squeeze(1)
        else:
            y_train_np = y_train_np
            y_val_np = y_val_np
            y_test_np = y_test_np

        val_aucs = []
        test_aucs = []
        models = []

        for i in range(y_test_np.shape[1]):
            if float(max(y_val_np[:, i])) == 0 or float(max(y_train_np[:, i])) == 0 or float(max(y_test_np[:, i])) == 0:
                continue

            train_valid_indices = np.where((y_train_np[:, i] == 0) | (y_train_np[:, i] == 1))[0]
            valid_valid_indices = np.where((y_val_np[:, i] == 0) | (y_val_np[:, i] == 1))[0]
            test_valid_indices = np.where((y_test_np[:, i] == 0) | (y_test_np[:, i] == 1))[0]

            model = XGBClassifier(
                base_score=0.5, booster='gbtree', colsample_bylevel=1,
                colsample_bytree=1, gamma=1, learning_rate=0.1, max_delta_step=0,
                max_depth=4, min_child_weight=8, missing=np.nan, n_estimators=2000,
                n_jobs=1, nthread=None, objective='binary:logistic', random_state=0,
                reg_alpha=0, reg_lambda=1, scale_pos_weight=1, seed=None,
                silent=True, subsample=0.8, tree_method='hist',
                eval_metric='auc', early_stopping_rounds=300
            )
            model.fit(X_train_np[train_valid_indices], y_train_np[:, i][train_valid_indices],
                     eval_set=[(X_val_np[valid_valid_indices], y_val_np[:, i][valid_valid_indices])])

            val_pred = model.predict_proba(X_val_np[valid_valid_indices])
            val_pred_proba = F.softmax(torch.tensor(val_pred), dim=-1)[:, 1].view(-1).numpy()
            val_auc = roc_auc_score(y_val_np[:, i][valid_valid_indices], val_pred_proba)
            val_aucs.append(val_auc)

            test_pred = model.predict_proba(X_test_np[test_valid_indices])
            test_pred_proba = F.softmax(torch.tensor(test_pred), dim=-1)[:, 1].view(-1).numpy()
            test_auc = roc_auc_score(y_test_np[:, i][test_valid_indices], test_pred_proba)
            test_aucs.append(test_auc)

            models.append(model)

        return np.mean(val_aucs), np.mean(test_aucs), models


def train_and_evaluate_rf_classification(cfg, X_train, y_train, X_val, y_val, X_test, y_test):
    """
    训练Random Forest分类模型，返回验证集分数和测试集预测
    """
    X_train_np = X_train.cpu().numpy()
    y_train_np = y_train.cpu().numpy()
    X_val_np = X_val.cpu().numpy()
    y_val_np = y_val.cpu().numpy()
    X_test_np = X_test.cpu().numpy()
    y_test_np = y_test.cpu().numpy()

    if y_test_np.shape[-1] == 1:
        y_train_np = y_train_np.ravel()
        y_val_np = y_val_np.ravel()
        y_test_np = y_test_np.ravel()

        model = RandomForestClassifier(
            n_estimators=200,
            max_depth=4,
            min_samples_split=8,
            n_jobs=-1,
            random_state=0
        )
        model.fit(X_train_np, y_train_np)

        val_pred = model.predict_proba(X_val_np)[:, 1]
        val_auc = roc_auc_score(y_val_np, val_pred)

        test_pred = model.predict_proba(X_test_np)[:, 1]
        test_auc = roc_auc_score(y_test_np, test_pred)

        return val_auc, test_auc, model
    else:
        if len(y_train_np.shape) == 3:
            y_train_np = y_train_np.squeeze(1)
            y_val_np = y_val_np.squeeze(1)
            y_test_np = y_test_np.squeeze(1)

        val_aucs = []
        test_aucs = []
        models = []

        for i in range(y_test_np.shape[1]):
            if float(max(y_val_np[:, i])) == 0 or float(max(y_train_np[:, i])) == 0 or float(max(y_test_np[:, i])) == 0:
                continue

            train_valid_indices = np.where((y_train_np[:, i] == 0) | (y_train_np[:, i] == 1))[0]
            valid_valid_indices = np.where((y_val_np[:, i] == 0) | (y_val_np[:, i] == 1))[0]
            test_valid_indices = np.where((y_test_np[:, i] == 0) | (y_test_np[:, i] == 1))[0]

            model = RandomForestClassifier(
                n_estimators=200,
                max_depth=4,
                min_samples_split=8,
                n_jobs=-1,
                random_state=0
            )
            model.fit(X_train_np[train_valid_indices], y_train_np[:, i][train_valid_indices])

            val_pred = model.predict_proba(X_val_np[valid_valid_indices])
            val_pred_proba = F.softmax(torch.tensor(val_pred), dim=-1)[:, 1].view(-1).numpy()
            val_auc = roc_auc_score(y_val_np[:, i][valid_valid_indices], val_pred_proba)
            val_aucs.append(val_auc)

            test_pred = model.predict_proba(X_test_np[test_valid_indices])
            test_pred_proba = F.softmax(torch.tensor(test_pred), dim=-1)[:, 1].view(-1).numpy()
            test_auc = roc_auc_score(y_test_np[:, i][test_valid_indices], test_pred_proba)
            test_aucs.append(test_auc)

            models.append(model)

        return np.mean(val_aucs), np.mean(test_aucs), models


def train_and_evaluate_lgb_classification(cfg, X_train, y_train, X_val, y_val, X_test, y_test):
    """
    训练LightGBM分类模型，返回验证集分数和测试集预测
    """
    X_train_np = X_train.detach().cpu().numpy()
    y_train_np = y_train.detach().cpu().numpy()
    X_val_np = X_val.detach().cpu().numpy()
    y_val_np = y_val.detach().cpu().numpy()
    X_test_np = X_test.detach().cpu().numpy()
    y_test_np = y_test.detach().cpu().numpy()

    if y_test_np.shape[-1] == 1:
        y_train_np = y_train_np.ravel()
        y_val_np = y_val_np.ravel()
        y_test_np = y_test_np.ravel()

        lgb_train = lgb.Dataset(X_train_np, y_train_np)
        lgb_eval = lgb.Dataset(X_val_np, y_val_np, reference=lgb_train)

        params = {
            'objective': 'binary',
            'metric': 'auc',
            'boosting_type': 'gbdt',
            'num_leaves': 31,
            'learning_rate': 0.05,
            'feature_fraction': 0.9,
            'bagging_fraction': 0.8,
            'bagging_freq': 5,
            'verbose': -1
        }

        callbacks = [lgb.early_stopping(stopping_rounds=300, verbose=False)]
        model = lgb.train(params, lgb_train, num_boost_round=2000,
                         valid_sets=[lgb_train, lgb_eval], callbacks=callbacks)

        val_pred = model.predict(X_val_np, num_iteration=model.best_iteration)
        val_auc = roc_auc_score(y_val_np, val_pred)

        test_pred = model.predict(X_test_np, num_iteration=model.best_iteration)
        test_auc = roc_auc_score(y_test_np, test_pred)

        return val_auc, test_auc, model
    else:
        val_aucs = []
        test_aucs = []
        models = []

        for i in range(y_test_np.shape[1]):
            if float(max(y_val_np[:, i])) == 0 or float(max(y_train_np[:, i])) == 0 or float(max(y_test_np[:, i])) == 0:
                continue

            train_valid_indices = np.where((y_train_np[:, i] == 0) | (y_train_np[:, i] == 1))[0]
            valid_valid_indices = np.where((y_val_np[:, i] == 0) | (y_val_np[:, i] == 1))[0]
            test_valid_indices = np.where((y_test_np[:, i] == 0) | (y_test_np[:, i] == 1))[0]

            lgb_train = lgb.Dataset(X_train_np[train_valid_indices], y_train_np[:, i][train_valid_indices])
            lgb_eval = lgb.Dataset(X_val_np[valid_valid_indices], y_val_np[:, i][valid_valid_indices], reference=lgb_train)

            params = {
                'objective': 'binary',
                'metric': 'auc',
                'boosting_type': 'gbdt',
                'num_leaves': 31,
                'learning_rate': 0.05,
                'feature_fraction': 0.9,
                'bagging_fraction': 0.8,
                'bagging_freq': 5,
                'verbose': -1
            }

            callbacks = [lgb.early_stopping(stopping_rounds=300, verbose=False)]
            model = lgb.train(params, lgb_train, num_boost_round=2000,
                            valid_sets=[lgb_train, lgb_eval], callbacks=callbacks)

            val_pred = model.predict(X_val_np[valid_valid_indices], num_iteration=model.best_iteration)
            val_auc = roc_auc_score(y_val_np[:, i][valid_valid_indices], val_pred)
            val_aucs.append(val_auc)

            test_pred = model.predict(X_test_np[test_valid_indices], num_iteration=model.best_iteration)
            test_auc = roc_auc_score(y_test_np[:, i][test_valid_indices], test_pred)
            test_aucs.append(test_auc)

            models.append(model)

        return np.mean(val_aucs), np.mean(test_aucs), models


def train_and_evaluate_xgb_regression(cfg, X_train, y_train, X_val, y_val, X_test, y_test, metric='rmse'):
    """
    训练XGBoost回归模型，返回验证集分数和测试集预测
    """
    X_train_np = X_train.cpu().numpy()
    y_train_np = y_train.cpu().numpy().ravel()
    X_val_np = X_val.cpu().numpy()
    y_val_np = y_val.cpu().numpy().ravel()
    X_test_np = X_test.cpu().numpy()
    y_test_np = y_test.cpu().numpy().ravel()

    if metric == 'rmse':
        model = XGBRegressor(
            learn_rate=0.1,
            max_depth=4,
            min_child_weight=10,
            gamma=1,
            subsample=0.8,
            colsample_bytree=0.8,
            reg_alpha=0.8,
            objective='reg:linear',
            n_estimators=2000,
            tree_method='hist',
            eval_metric='rmse',
            early_stopping_rounds=300
        )
    else:
        model = XGBRegressor(
            objective='reg:squarederror',
            eval_metric='mae',
            learn_rate=0.1,
            max_depth=4,
            min_child_weight=10,
            gamma=1,
            subsample=0.8,
            colsample_bytree=0.8,
            reg_alpha=0.8,
            n_estimators=2000,
            tree_method='hist',
            early_stopping_rounds=300
        )

    model.fit(X_train_np, y_train_np, eval_set=[(X_val_np, y_val_np)], verbose=False)

    val_pred = model.predict(X_val_np)
    test_pred = model.predict(X_test_np)

    val_pred = val_pred.ravel()
    test_pred = test_pred.ravel()

    if metric == 'rmse':
        val_score = mean_squared_error(y_val_np, val_pred) ** 0.5
        test_score = mean_squared_error(y_test_np, test_pred) ** 0.5
    else:
        val_score = mean_absolute_error(y_val_np, val_pred)
        test_score = mean_absolute_error(y_test_np, test_pred)

    return val_score, test_score, model


def train_and_evaluate_rf_regression(cfg, X_train, y_train, X_val, y_val, X_test, y_test, metric='rmse'):
    """
    训练Random Forest回归模型，返回验证集分数和测试集预测
    """
    X_train_np = X_train.cpu().numpy()
    X_val_np = X_val.cpu().numpy()
    X_test_np = X_test.cpu().numpy()
    y_train_np = y_train.cpu().numpy()
    y_val_np = y_val.cpu().numpy()
    y_test_np = y_test.cpu().numpy()

    if len(y_train_np.shape) == 1 or y_train_np.shape[1] == 1:
        y_train_np = y_train_np.ravel()
        y_val_np = y_val_np.ravel()
        y_test_np = y_test_np.ravel()

    y_original_dim = y_train.shape[-1] if len(y_train.shape) > 1 else 1

    if y_original_dim == 1:
        model = RandomForestRegressor(
            n_estimators=200,
            max_depth=8,
            min_samples_split=10,
            min_samples_leaf=4,
            max_features='log2',
            bootstrap=True,
            oob_score=False,
            random_state=42,
            n_jobs=-1
        )
        model.fit(X_train_np, y_train_np)

        val_pred = model.predict(X_val_np)
        test_pred = model.predict(X_test_np)

        val_pred = val_pred.ravel()
        test_pred = test_pred.ravel()

        if metric == 'rmse':
            val_score = mean_squared_error(y_val_np, val_pred) ** 0.5
            test_score = mean_squared_error(y_test_np, test_pred) ** 0.5
        else:
            val_score = mean_absolute_error(y_val_np, val_pred)
            test_score = mean_absolute_error(y_test_np, test_pred)

        return val_score, test_score, model
    else:
        val_scores = []
        test_scores = []
        models = []

        for i in range(y_test_np.shape[1]):
            if np.sum(~np.isnan(y_train_np[:, i])) == 0 or np.sum(~np.isnan(y_val_np[:, i])) == 0 or np.sum(~np.isnan(y_test_np[:, i])) == 0:
                continue

            train_valid_indices = np.where(~np.isnan(y_train_np[:, i]))[0]
            valid_valid_indices = np.where(~np.isnan(y_val_np[:, i]))[0]
            test_valid_indices = np.where(~np.isnan(y_test_np[:, i]))[0]

            model = RandomForestRegressor(
                n_estimators=200,
                max_depth=8,
                min_samples_split=10,
                min_samples_leaf=4,
                max_features='log2',
                bootstrap=True,
                oob_score=False,
                random_state=42,
                n_jobs=-1
            )
            model.fit(X_train_np[train_valid_indices], y_train_np[:, i][train_valid_indices])

            if metric == 'rmse':
                val_pred = model.predict(X_val_np[valid_valid_indices])
                val_score = mean_squared_error(y_val_np[:, i][valid_valid_indices], val_pred) ** 0.5
                test_pred = model.predict(X_test_np[test_valid_indices])
                test_score = mean_squared_error(y_test_np[:, i][test_valid_indices], test_pred) ** 0.5
            else:
                val_pred = model.predict(X_val_np[valid_valid_indices])
                val_score = mean_absolute_error(y_val_np[:, i][valid_valid_indices], val_pred)
                test_pred = model.predict(X_test_np[test_valid_indices])
                test_score = mean_absolute_error(y_test_np[:, i][test_valid_indices], test_pred)

            val_scores.append(val_score)
            test_scores.append(test_score)
            models.append(model)

        return np.mean(val_scores), np.mean(test_scores), models


def train_and_evaluate_lgb_regression(cfg, X_train, y_train, X_val, y_val, X_test, y_test, metric='rmse'):
    """
    训练LightGBM回归模型，返回验证集分数和测试集预测
    """
    X_train_np = X_train.detach().cpu().numpy()
    y_train_np = y_train.detach().cpu().numpy()
    X_val_np = X_val.detach().cpu().numpy()
    y_val_np = y_val.detach().cpu().numpy()
    X_test_np = X_test.detach().cpu().numpy()
    y_test_np = y_test.detach().cpu().numpy()

    y_original_dim = y_train.shape[-1] if len(y_train.shape) > 1 else 1

    if len(y_train_np.shape) == 1 or y_train_np.shape[1] == 1:
        y_train_np = y_train_np.ravel()
        y_val_np = y_val_np.ravel()
        y_test_np = y_test_np.ravel()

    if y_original_dim == 1:
        lgb_train = lgb.Dataset(X_train_np, y_train_np)
        lgb_eval = lgb.Dataset(X_val_np, y_val_np, reference=lgb_train)

        if metric == 'rmse':
            params = {
                'objective': 'regression',
                'metric': 'rmse',
                'boosting_type': 'gbdt',
                'num_leaves': 63,
                'max_depth': 9,
                'learning_rate': 0.02,
                'feature_fraction': 0.9,
                'bagging_fraction': 0.7,
                'bagging_freq': 5,
                'min_data_in_leaf': 5,
                'lambda_l1': 1e-5,
                'lambda_l2': 0.1,
                'min_gain_to_split': 0.1,
                'random_state': 42
            }
        else:
            params = {
                'objective': 'regression',
                'metric': 'mae',
                'boosting_type': 'gbdt',
                'num_leaves': 63,
                'max_depth': 9,
                'learning_rate': 0.02,
                'feature_fraction': 0.9,
                'bagging_fraction': 0.7,
                'bagging_freq': 5,
                'min_data_in_leaf': 5,
                'lambda_l1': 1e-5,
                'lambda_l2': 0.1,
                'min_gain_to_split': 0.1,
                'verbose': -1,
                'random_state': 42
            }

        callbacks = [lgb.early_stopping(stopping_rounds=50)]
        model = lgb.train(params, lgb_train, num_boost_round=2000,
                         valid_sets=[lgb_train, lgb_eval], callbacks=callbacks)

        val_pred = model.predict(X_val_np, num_iteration=model.best_iteration)
        test_pred = model.predict(X_test_np, num_iteration=model.best_iteration)

        val_pred = val_pred.ravel()
        test_pred = test_pred.ravel()

        if metric == 'rmse':
            val_score = mean_squared_error(y_val_np, val_pred) ** 0.5
            test_score = mean_squared_error(y_test_np, test_pred) ** 0.5
        else:
            val_score = mean_absolute_error(y_val_np, val_pred)
            test_score = mean_absolute_error(y_test_np, test_pred)

        return val_score, test_score, model
    else:
        val_scores = []
        test_scores = []
        models = []

        for i in range(y_test_np.shape[1]):
            if np.sum(~np.isnan(y_train_np[:, i])) == 0 or np.sum(~np.isnan(y_val_np[:, i])) == 0 or np.sum(~np.isnan(y_test_np[:, i])) == 0:
                continue

            train_valid_indices = np.where(~np.isnan(y_train_np[:, i]))[0]
            valid_valid_indices = np.where(~np.isnan(y_val_np[:, i]))[0]
            test_valid_indices = np.where(~np.isnan(y_test_np[:, i]))[0]

            lgb_train = lgb.Dataset(X_train_np[train_valid_indices], y_train_np[:, i][train_valid_indices])
            lgb_eval = lgb.Dataset(X_val_np[valid_valid_indices], y_val_np[:, i][valid_valid_indices], reference=lgb_train)

            if metric == 'rmse':
                params = {
                    'objective': 'regression',
                    'metric': 'rmse',
                    'boosting_type': 'gbdt',
                    'num_leaves': 31,
                    'max_depth': 6,
                    'learning_rate': 0.05,
                    'feature_fraction': 0.8,
                    'bagging_fraction': 0.8,
                    'bagging_freq': 1,
                    'min_data_in_leaf': 20,
                    'lambda_l1': 1e-5,
                    'lambda_l2': 1,
                    'verbose': -1,
                    'random_state': 42
                }
            else:
                params = {
                    'objective': 'regression',
                    'metric': 'mae',
                    'boosting_type': 'gbdt',
                    'num_leaves': 31,
                    'max_depth': 6,
                    'learning_rate': 0.05,
                    'feature_fraction': 0.8,
                    'bagging_fraction': 0.8,
                    'bagging_freq': 1,
                    'min_data_in_leaf': 20,
                    'lambda_l1': 1e-5,
                    'lambda_l2': 1,
                    'verbose': -1,
                    'random_state': 42
                }

            callbacks = [lgb.early_stopping(stopping_rounds=150, verbose=False)]
            model = lgb.train(params, lgb_train, num_boost_round=1000,
                            valid_sets=[lgb_train, lgb_eval], callbacks=callbacks)

            if metric == 'rmse':
                val_pred = model.predict(X_val_np[valid_valid_indices], num_iteration=model.best_iteration)
                val_score = mean_squared_error(y_val_np[:, i][valid_valid_indices], val_pred) ** 0.5
                test_pred = model.predict(X_test_np[test_valid_indices], num_iteration=model.best_iteration)
                test_score = mean_squared_error(y_test_np[:, i][test_valid_indices], test_pred) ** 0.5
            else:
                val_pred = model.predict(X_val_np[valid_valid_indices], num_iteration=model.best_iteration)
                val_score = mean_absolute_error(y_val_np[:, i][valid_valid_indices], val_pred)
                test_pred = model.predict(X_test_np[test_valid_indices], num_iteration=model.best_iteration)
                test_score = mean_absolute_error(y_test_np[:, i][test_valid_indices], test_pred)

            val_scores.append(val_score)
            test_scores.append(test_score)
            models.append(model)

        return np.mean(val_scores), np.mean(test_scores), models


def select_best_model_and_predict(cfg, train_features, train_targets, val_features, val_targets, test_features, test_targets, logger):

    logger.info("=" * 50)
    logger.info("基于验证集选择最优模型...")
    logger.info("=" * 50)

    if cfg.DATA.TASK_TYPE == 'classification':
        logger.info("任务类型: 分类 (ROC-AUC)")

        xgb_val_score, xgb_test_score, xgb_model = train_and_evaluate_xgb_classification(
            cfg, train_features, train_targets, val_features, val_targets, test_features, test_targets)

        rf_val_score, rf_test_score, rf_model = train_and_evaluate_rf_classification(
            cfg, train_features, train_targets, val_features, val_targets, test_features, test_targets)

        lgb_val_score, lgb_test_score, lgb_model = train_and_evaluate_lgb_classification(
            cfg, train_features, train_targets, val_features, val_targets, test_features, test_targets)

        val_scores = {'xgb': xgb_val_score, 'rf': rf_val_score, 'lgb': lgb_val_score}
        test_scores = {'xgb': xgb_test_score, 'rf': rf_test_score, 'lgb': lgb_test_score}
        models = {'xgb': xgb_model, 'rf': rf_model, 'lgb': lgb_model}

    else:
        metric = cfg.DATA.METRIC
        logger.info(f"任务类型: 回归 ({metric})")

        xgb_val_score, xgb_test_score, xgb_model = train_and_evaluate_xgb_regression(
            cfg, train_features, train_targets, val_features, val_targets, test_features, test_targets, metric)

        rf_val_score, rf_test_score, rf_model = train_and_evaluate_rf_regression(
            cfg, train_features, train_targets, val_features, val_targets, test_features, test_targets, metric)

        lgb_val_score, lgb_test_score, lgb_model = train_and_evaluate_lgb_regression(
            cfg, train_features, train_targets, val_features, val_targets, test_features, test_targets, metric)

        val_scores = {'xgb': xgb_val_score, 'rf': rf_val_score, 'lgb': lgb_val_score}
        test_scores = {'xgb': xgb_test_score, 'rf': rf_test_score, 'lgb': lgb_test_score}
        models = {'xgb': xgb_model, 'rf': rf_model, 'lgb': lgb_model}

    if cfg.DATA.TASK_TYPE == 'classification':
        best_model_name = max(val_scores, key=val_scores.get)
    else:
        best_model_name = min(val_scores, key=val_scores.get)

    best_val_score = val_scores[best_model_name]
    best_test_score = test_scores[best_model_name]

    return best_model_name, best_val_score, best_test_score