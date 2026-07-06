"""
    数据增强：
        1. 读取csv文件
        2. 按行对smiles字符串进行增强
        3. 另存为csv文件
"""
# 1.先读取csv文件
import pandas as pd
import numpy as np
from smiles_enumerator import SmilesEnumerator
sme = SmilesEnumerator()


def list_to_dict(smiles_list, property_dict):
    """
    将列表转换为字典，键为'smiles'，值为列表中的数据，'activity'默认为1。

    :param smiles_list: 包含smiles字符串的列表
    :return: 字典，包含'smiles'和'activity'
    """
    result = []
    for smiles in smiles_list:
        result.append({
            'smiles': smiles,
            **property_dict     # 字典解包
        })
    return result


def randomize_smiles(smiles_str, random_num):
    # 生成random_num个smiles
    smiles_list = {smiles_str}  # 用集合来去重
    i = 0   # 记录生成次数
    while len(smiles_list) < random_num and i < random_num * 2:  # 控制最大生成次数
        random_smiles_str = sme.randomize_smiles(smiles_str)
        smiles_list.add(random_smiles_str)
        i += 1
    return list(smiles_list)


# def data_enhancement(df, random_num=10):
#     data_result = []
#     # 记录所有扩增后的数据[{"smiles": "balabalba1", "activity": 1},...,{{"smiles": "balabalba2", "activity": 0}}]
#     # 逐行处理CSV文件
#     for index, row in df.iterrows():
#         smiles = row['smiles']
#         activity = row['activity']
#         smiles_list = randomize_smiles(smiles, random_num)  # 记录生成的random_num个smiles表达式
#         smiles_activity = list_to_dict(smiles_list, activity)   # 给每个smiles添加活性
#         data_result.extend(smiles_activity) # 汇总所有扩增的数据
#     # 保存到文件
#     df_output = pd.DataFrame(data_result)
#     # df_output.to_csv(output_name, index=False)
#     return df_output

def data_enhancement(df, random_num, seed=None):
    """
    数据增强函数，支持固定 seed 使结果可复现。
    
    :param df: 输入的 DataFrame
    :param random_num: 每个分子增强的 SMILES 数量
    :param seed: 随机种子，用于固定 numpy 随机状态。如果为 None，则不固定（保持可复现但不跨进程同步）
    """
    if random_num == 1:
        return df
    
    # 如果提供了 seed，设置 numpy 全局随机状态
    if seed is not None:
        np.random.seed(seed)
    
    data_result = []
    # 记录所有扩增后的数据[{"smiles": "balabalba1", "activity": 1},...,{{"smiles": "balabalba2", "activity": 0}}]
    # 逐行处理CSV文件
    from rdkit import Chem
    from rdkit import RDLogger
    RDLogger.DisableLog('rdApp.*')
    skipped = 0
    
    for idx, (index, row) in enumerate(df.iterrows()):
        # 每行使用不同的 seed，确保增强结果与处理顺序无关
        if seed is not None:
            np.random.seed(seed + idx)
        
        smiles = row['smiles']
        smiles_list = randomize_smiles(smiles, random_num)  # 记录生成的random_num个smiles表达式
        # Keep the DIVERSE randomized SMILES (validate only). Canonicalizing here
        # would collapse every variant back to one string -> pure duplication and
        # zero augmentation. The paper's augmentation relies on diverse SMILES
        # traversals, so we keep distinct valid strings.
        valid_smiles = []
        for s in smiles_list:
            try:
                if Chem.MolFromSmiles(s) is not None:
                    valid_smiles.append(s)
            except Exception:
                skipped += 1
        # de-dup identical strings while preserving order
        seen = set()
        valid_smiles = [x for x in valid_smiles if not (x in seen or seen.add(x))]
        if not valid_smiles:
            skipped += 1
            continue
        property_dict = {}
        for col in df.columns:
            if col != 'smiles':
                col_value = row[col]
                property_dict[col] = col_value

        smiles_property = list_to_dict(valid_smiles, property_dict)   # 给每个smiles添加活性
        data_result.extend(smiles_property) # 汇总所有扩增的数据

    if skipped > 0:
        print(f'WARNING: skipped {skipped} invalid SMILES during augmentation')
    df_output = pd.DataFrame(data_result)
    # df_output.to_csv(output_name, index=False)
    return df_output



def main(csv_name, output_name, random_num=10):
    data_result = []
    # 记录所有扩增后的数据[{"smiles": "balabalba1", "activity": 1},...,{{"smiles": "balabalba2", "activity": 0}}]
    df = pd.read_csv(csv_name)
    # 逐行处理CSV文件
    for index, row in df.iterrows():
        smiles = row['smiles']
        activity = row['activity']
        smiles_list = randomize_smiles(smiles, random_num)  # 记录生成的random_num个smiles表达式
        smiles_activity = list_to_dict(smiles_list, activity)   # 给每个smiles添加活性
        data_result.extend(smiles_activity) # 汇总所有扩增的数据
    # 保存到文件
    df_output = pd.DataFrame(data_result)
    df_output.to_csv(output_name, index=False) 
    return None


if __name__ == "__main__":

    file = "nottianran"
    random_num = 5
    csv_name = f"{file}.csv"
    output_csv = f"{file}_x_{random_num}.csv"
    main(csv_name, output_csv, random_num)


