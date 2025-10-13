# toolkit/split_utils.py
from pathlib import Path
import random
from collections import defaultdict
from typing import List, Tuple

def _group_key(p: str, mode: str) -> str:
    """
    根据 mode 决定“哪些文件属于同一组”。

    参数：
    p: 文件路径，比如 'data/F1_train_001.xlsx'
    mode: 分组方式，有三种选择
          - "file":  每个文件单独作为一组（最严格，不同文件一定被分开）
          - "prefix": 根据文件名前缀（第一个'_'之前的部分）分组
          - "parent": 根据上一级文件夹的名字分组

    返回值：
    这个函数会返回一个“组名”，比如：
      如果 mode='prefix' 且文件名是 'A001_2025.xlsx'，返回 'A001'
    """
    path = Path(p)  # 把字符串路径转成 Path 对象，方便处理
    if mode == "file":
        return path.stem  # 文件名（不带后缀）作为组ID
    if mode == "prefix":
        return path.stem.split('_')[0]  # 取 '_' 前面的部分
    if mode == "parent":
        return path.parent.name  # 上一级文件夹名
    return path.stem  # 默认按文件粒度

def grouped_split(
    file_paths: List[str],
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
    seed: int = 42,
    group_by: str = "file",
) -> Tuple[List[str], List[str], List[str]]:
    """
    把文件路径列表，按比例切分成 3 个集合：train、val、test。

    参数：
    file_paths : 所有文件路径的列表
    train_ratio: 训练集占比
    val_ratio  : 验证集占比
    seed       : 随机种子（保证每次划分结果一致）
    group_by   : 分组依据，和上面的 _group_key 对应

    返回值：
    (train_files, val_files, test_files) 三个列表
    """
    random.seed(seed)  # 设置随机种子，保证可复现
    buckets = defaultdict(list)  # 创建一个字典：key=组ID, value=该组的文件列表

    # --- 第一步：按“组”把文件分类 ---
    for p in file_paths:
        gid = _group_key(p, group_by)  # 获取这个文件的组ID
        buckets[gid].append(p)         # 把它放到对应组里

    # --- 第二步：打乱组的顺序（而不是打乱文件）---
    gids = list(buckets.keys())  # 所有组的名字
    random.shuffle(gids)         # 打乱组顺序，让划分更随机

    # --- 第三步：根据比例计算每部分多少组 ---
    n = len(gids)
    n_train = int(n * train_ratio)  # 训练集组数
    n_val = int(n * val_ratio)      # 验证集组数

    # --- 第四步：取前几组作为 train，接着是 val，剩下的是 test ---
    train_ids = set(gids[:n_train])
    val_ids   = set(gids[n_train:n_train+n_val])
    test_ids  = set(gids[n_train+n_val:])

    # --- 第五步：把每个组的文件展开成最终列表 ---
    def pick(ids):
        out = []
        for gid in ids:
            out.extend(buckets[gid])  # 把该组的所有文件加入结果
        return sorted(out)  # 排序，让输出更整齐

    return pick(train_ids), pick(val_ids), pick(test_ids)
