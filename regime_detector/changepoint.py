# -*- coding: utf-8 -*-
"""
变点检测模块 (changepoint.py)
==============================
实现多种变点检测算法，识别市场收益率均值/方差的结构性变化。

算法包括：
    1. CUSUM（累积和）变点检测
    2. PELT算法（ruptures库）
    3. 贝叶斯在线变点检测（简化版）

所有检测均基于真实市场数据，不引入任何随机或伪造数据。
"""

import numpy as np
import pandas as pd
import ruptures as rpt
from typing import List, Dict, Optional, Tuple


class ChangepointDetector:
    """变点检测器，支持CUSUM、PELT和贝叶斯在线检测。"""

    def __init__(self, threshold: float = 3.0, min_size: int = 20):
        """
        初始化变点检测器。

        参数:
            threshold: CUSUM检测阈值（标准差倍数）
            min_size: 变点间最小间隔（样本数）
        """
        self.threshold = threshold
        self.min_size = min_size

    # ========================
    # CUSUM 变点检测
    # ========================

    def cusum_detect(self, data: np.ndarray,
                     threshold: Optional[float] = None) -> List[Dict]:
        """
        CUSUM（累积和）变点检测。

        CUSUM通过累积偏离均值的偏差来检测均值漂移。
        当累积和超过阈值时，判定为变点。

        算法步骤:
            1. 计算序列均值
            2. 计算上下累积和 S+ 和 S-
            3. 当 S+ 或 S- 超过阈值时，记录变点并重置累积和

        参数:
            data: 一维数据序列（通常为收益率）
            threshold: 检测阈值（标准差倍数），None则使用self.threshold

        返回:
            变点列表，每个元素包含位置和置信度
        """
        if threshold is None:
            threshold = self.threshold

        data = np.asarray(data, dtype=float)
        n = len(data)

        if n < 2 * self.min_size:
            return []

        # 计算全局均值和标准差
        mu = data.mean()
        sigma = data.std()

        if sigma == 0:
            return []

        # 标准化阈值
        h = threshold * sigma

        # CUSUM累积和
        s_pos = 0.0  # 正向累积和（检测上升变点）
        s_neg = 0.0  # 负向累积和（检测下降变点）

        changepoints = []
        last_cp = 0

        for i in range(n):
            deviation = data[i] - mu
            s_pos = max(0, s_pos + deviation)
            s_neg = min(0, s_neg + deviation)

            # 检测变点
            if s_pos > h or abs(s_neg) > h:
                # 确保变点间距
                if i - last_cp >= self.min_size:
                    # 置信度基于超出阈值的程度
                    confidence = min(1.0, max(s_pos, abs(s_neg)) / (h * 2))
                    changepoints.append({
                        "位置": i,
                        "日期索引": i,
                        "类型": "上升" if s_pos > h else "下降",
                        "CUSUM值": float(max(s_pos, abs(s_neg))),
                        "置信度": float(confidence),
                        "算法": "CUSUM",
                    })
                    last_cp = i

                # 重置累积和
                s_pos = 0.0
                s_neg = 0.0

        return changepoints

    # ========================
    # PELT 变点检测 (ruptures库)
    # ========================

    def pelt_detect(self, data: np.ndarray,
                    model: str = "rbf",
                    pen: Optional[float] = None) -> List[Dict]:
        """
        PELT（Pruned Exact Linear Time）变点检测。

        PELT是一种精确的变点检测算法，时间复杂度为O(n)。
        通过最小化代价函数+惩罚项来寻找最优变点集。

        支持的模型（代价函数）:
            - "l1": 均值变化（L1范数）
            - "l2": 均值变化（L2范数）
            - "rbf": 均值和方差变化（径向基核）
            - "normal": 正态分布参数变化
            - "ar": 自回归模型参数变化

        参数:
            data: 数据序列（一维或多维）
            model: 代价函数模型
            pen: 惩罚参数，None则自动估计

        返回:
            变点列表，每个元素包含位置和置信度
        """
        data = np.asarray(data, dtype=float)

        # 如果是一维数据，reshape为二维
        if data.ndim == 1:
            signal = data.reshape(-1, 1)
        else:
            signal = data

        n = len(signal)
        if n < 2 * self.min_size:
            return []

        # 自动估计惩罚参数
        if pen is None:
            # 使用BIC准则估计惩罚
            # pen = n * log(n) / (n^0.5) 的经验值
            pen = np.log(n) * signal.var()

        try:
            # 使用ruptures的PELT算法
            algo = rpt.Pelt(model=model, min_size=self.min_size).fit(signal)
            bkps = algo.predict(pen=pen)
        except Exception as e:
            # 如果PELT失败，尝试使用Dynp作为后备
            try:
                n_bkps = max(1, n // (self.min_size * 3))
                algo = rpt.Dynp(model=model, min_size=self.min_size).fit(signal)
                bkps = algo.predict(n_bkps=n_bkps)
            except Exception:
                return []

        # 转换为变点列表（ruptures返回的最后一个元素是序列末尾n）
        changepoints = []
        for cp in bkps[:-1]:  # 排除末尾
            if cp > 0 and cp < n:
                # 计算变点前后的统计差异作为置信度
                before = signal[:cp]
                after = signal[cp:]

                if len(before) > 0 and len(after) > 0:
                    # 均值差异的标准化程度
                    mean_diff = abs(before.mean() - after.mean())
                    std_combined = np.sqrt(
                        (before.var() + after.var()) / 2 + 1e-10
                    )
                    confidence = min(1.0, mean_diff / (std_combined + 1e-10))

                    changepoints.append({
                        "位置": cp,
                        "日期索引": cp,
                        "类型": "结构性变化",
                        "前后均值差": float(mean_diff),
                        "置信度": float(confidence),
                        "算法": "PELT",
                        "模型": model,
                    })

        return changepoints

    # ========================
    # 贝叶斯在线变点检测（简化版）
    # ========================

    def bayesian_online_detect(self, data: np.ndarray,
                               hazard_rate: float = 0.01,
                               max_run_length: int = 500) -> List[Dict]:
        """
        贝叶斯在线变点检测（简化版）。

        基于Adams & MacKay (2007)的算法，通过维护运行长度
        （run-length）的后验分布来在线检测变点。

        算法核心:
            - 维护运行长度r的预测概率P(r_t | x_1:t)
            - 每个时间步更新运行长度分布
            - 当运行长度为0的概率突增时，判定为变点

        参数:
            data: 一维数据序列
            hazard_rate: 危险率（变点先验概率），默认1%
            max_run_length: 最大运行长度（截断计算）

        返回:
            变点列表
        """
        data = np.asarray(data, dtype=float)
        n = len(data)

        if n < 2 * self.min_size:
            return []

        # 初始化运行长度概率分布
        # R[t] = P(r_t = t | x_1:t)，运行长度为t的概率
        R = np.zeros((n, max_run_length + 1))
        R[0, 0] = 1.0  # 初始运行长度为0

        # 使用简单的正态分布作为预测模型
        # 逐步更新均值和方差估计
        mu = 0.0  # 先验均值
        kappa = 1.0  # 先验精度
        alpha = 1.0  # 先验形状参数
        beta = 1.0  # 先验尺度参数

        # 存储变点概率
        changepoint_probs = np.zeros(n)

        for t in range(1, n):
            # 计算预测概率（学生t分布近似为正态）
            x = data[t]

            # 对每个可能的运行长度计算预测概率
            # 简化：使用滑动窗口的均值和方差
            max_r = min(t, max_run_length)

            for r in range(max_r + 1):
                # 当前运行长度为r时，预测概率
                if r == 0:
                    # 新运行长度，使用先验
                    pred_prob = self._normal_pdf(x, mu, beta / alpha)
                else:
                    # 使用历史数据估计参数
                    start_idx = max(0, t - r)
                    window_data = data[start_idx:t]
                    if len(window_data) > 0:
                        w_mean = window_data.mean()
                        w_std = window_data.std() + 1e-8
                        pred_prob = self._normal_pdf(x, w_mean, w_std)
                    else:
                        pred_prob = self._normal_pdf(x, mu, beta / alpha)

                # 危险函数 H(r) = 1/lambda (常数危险率)
                H = hazard_rate

                # 运行长度增长概率: (1-H) * R[t-1, r-1] * pred_prob
                if r > 0 and t > 0:
                    growth_prob = (1 - H) * R[t - 1, r - 1] * pred_prob
                else:
                    growth_prob = 0

                # 新变点概率: H * sum(R[t-1, :]) * pred_prob
                if r == 0:
                    cp_prob = H * R[t - 1, :max_r + 1].sum() * pred_prob
                else:
                    cp_prob = 0

                R[t, r] = growth_prob + cp_prob

            # 归一化
            total = R[t, :max_r + 1].sum()
            if total > 0:
                R[t, :max_r + 1] /= total

            # 变点概率 = P(r_t = 0 | x_1:t)
            changepoint_probs[t] = R[t, 0]

        # 检测变点：变点概率超过阈值且局部最大值
        changepoints = []
        cp_threshold = 0.3  # 变点概率阈值
        last_cp = 0

        for t in range(self.min_size, n):
            if changepoint_probs[t] > cp_threshold:
                # 检查是否为局部最大值
                is_local_max = True
                window = 5
                for j in range(max(0, t - window), min(n, t + window + 1)):
                    if j != t and changepoint_probs[j] > changepoint_probs[t]:
                        is_local_max = False
                        break

                if is_local_max and (t - last_cp) >= self.min_size:
                    changepoints.append({
                        "位置": t,
                        "日期索引": t,
                        "类型": "贝叶斯变点",
                        "变点概率": float(changepoint_probs[t]),
                        "置信度": float(changepoint_probs[t]),
                        "算法": "Bayesian-Online",
                    })
                    last_cp = t

        return changepoints

    def _normal_pdf(self, x: float, mean: float, std: float) -> float:
        """计算正态分布概率密度函数值。"""
        if std <= 0:
            std = 1e-8
        return float(
            np.exp(-0.5 * ((x - mean) / std) ** 2) / (std * np.sqrt(2 * np.pi))
        )

    # ========================
    # 综合变点检测
    # ========================

    def detect_all(self, data: np.ndarray,
                   methods: Optional[List[str]] = None) -> Dict:
        """
        使用多种算法综合检测变点。

        参数:
            data: 数据序列
            methods: 使用的算法列表，None表示全部

        返回:
            包含各算法结果的字典
        """
        if methods is None:
            methods = ["cusum", "pelt", "bayesian"]

        results = {}

        if "cusum" in methods:
            results["cusum"] = self.cusum_detect(data)

        if "pelt" in methods:
            results["pelt"] = self.pelt_detect(data, model="rbf")

        if "bayesian" in methods:
            results["bayesian"] = self.bayesian_online_detect(data)

        # 合并变点（去重）
        all_cps = []
        for method, cps in results.items():
            all_cps.extend(cps)

        # 按位置排序
        all_cps.sort(key=lambda x: x["位置"])

        # 合并相近的变点（距离小于min_size的合并）
        merged_cps = []
        for cp in all_cps:
            if not merged_cps or \
               cp["位置"] - merged_cps[-1]["位置"] >= self.min_size:
                merged_cps.append(cp)
            else:
                # 保留置信度更高的
                if cp["置信度"] > merged_cps[-1]["置信度"]:
                    merged_cps[-1] = cp

        results["merged"] = merged_cps
        results["summary"] = {
            "总变点数": len(merged_cps),
            "CUSUM变点": len(results.get("cusum", [])),
            "PELT变点": len(results.get("pelt", [])),
            "贝叶斯变点": len(results.get("bayesian", [])),
        }

        return results

    def detect_mean_change(self, data: np.ndarray) -> List[Dict]:
        """专门检测均值变化（使用PELT l2模型）。"""
        return self.pelt_detect(data, model="l2")

    def detect_variance_change(self, data: np.ndarray) -> List[Dict]:
        """专门检测方差变化（使用PELT rbf模型）。"""
        return self.pelt_detect(data, model="rbf")
