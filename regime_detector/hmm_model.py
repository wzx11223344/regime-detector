# -*- coding: utf-8 -*-
"""
HMM隐马尔可夫模型模块 (hmm_model.py)
=====================================
使用hmmlearn实现高斯HMM进行市场状态检测。
通过Baum-Welch算法训练模型，Viterbi算法解码最优状态序列。

支持2-4状态市场状态识别：
    - 牛市(Bull): 高正收益、低波动
    - 熊市(Bear): 负收益、高波动
    - 震荡(Range): 接近零收益、中低波动
    - 危机(Crisis): 大幅负收益、极高波动

注意：HMM初始化使用random_state=42保证可复现性，不用于生成业务数据。
"""

import numpy as np
import pandas as pd
from hmmlearn.hmm import GaussianHMM
from typing import Optional, List, Dict, Tuple


class HMMRegimeDetector:
    """基于高斯HMM的市场状态检测器。"""

    # 状态标签映射（按状态数量动态选择）
    STATE_LABELS = {
        2: ["熊市", "牛市"],
        3: ["熊市", "震荡", "牛市"],
        4: ["危机", "熊市", "震荡", "牛市"],
    }

    # 状态颜色映射（用于可视化）
    STATE_COLORS = {
        "牛市": "#2ecc71",      # 绿色
        "震荡": "#f39c12",      # 橙色
        "熊市": "#e74c3c",      # 红色
        "危机": "#8b0000",      # 暗红
    }

    def __init__(self, n_states: int = 3, n_iter: int = 100,
                 random_state: int = 42):
        """
        初始化HMM市场状态检测器。

        参数:
            n_states: 隐状态数量，2-4
            n_iter: Baum-Welch算法最大迭代次数
            random_state: 随机种子，保证结果可复现（仅用于HMM初始化）
        """
        if n_states not in [2, 3, 4]:
            raise ValueError(f"状态数必须为2/3/4，当前: {n_states}")

        self.n_states = n_states
        self.n_iter = n_iter
        self.random_state = random_state  # 保证可复现，非业务数据生成
        self.model: Optional[GaussianHMM] = None
        self.state_labels: List[str] = []
        self.state_order: Optional[np.ndarray] = None
        self.feature_names: List[str] = []

    def fit(self, X: np.ndarray,
            feature_names: Optional[List[str]] = None) -> "HMMRegimeDetector":
        """
        使用Baum-Welch算法训练HMM模型。

        Baum-Welch是EM算法在HMM上的特例，通过最大化观测序列的
        对数似然来估计模型参数（转移概率矩阵、发射概率参数）。

        参数:
            X: 特征矩阵，shape=(n_samples, n_features)
            feature_names: 特征名列表

        返回:
            self（支持链式调用）
        """
        if feature_names is not None:
            self.feature_names = feature_names

        # 创建高斯HMM模型
        # covariance_type="full" 允许特征间相关性
        # random_state=42 保证初始化可复现
        self.model = GaussianHMM(
            n_components=self.n_states,
            covariance_type="full",
            n_iter=self.n_iter,
            random_state=self.random_state,
            tol=1e-4,  # 收敛阈值
        )

        # Baum-Welch算法训练（EM算法）
        self.model.fit(X)

        # 根据第一个特征（通常是log_return）的均值对状态排序
        # 均值越高的状态对应越"牛"的市场
        self._sort_states_by_mean()

        return self

    def _sort_states_by_mean(self):
        """
        根据状态均值对状态重新排序。

        HMM训练后的状态编号是任意的，需要根据经济含义重新排序。
        通常按第一个特征（对数收益率）的均值排序：
            均值最高 -> 牛市
            均值最低 -> 熊市/危机
        """
        # 获取各状态在第一个特征上的均值
        means = self.model.means_[:, 0]
        # 按均值从小到大排序（熊市在前，牛市在后）
        self.state_order = np.argsort(means)

        # 分配状态标签
        self.state_labels = self.STATE_LABELS[self.n_states]

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        使用Viterbi算法解码最优隐状态序列。

        Viterbi算法通过动态规划找到最可能产生观测序列的
        隐状态路径，是最优状态序列的ML估计。

        参数:
            X: 特征矩阵

        返回:
            状态序列数组，值域 [0, n_states-1]（已排序）
        """
        if self.model is None:
            raise RuntimeError("模型未训练，请先调用fit()")

        # Viterbi解码
        raw_states = self.model.predict(X)

        # 将原始状态映射到排序后的状态
        sorted_states = np.zeros_like(raw_states)
        for new_idx, old_idx in enumerate(self.state_order):
            sorted_states[raw_states == old_idx] = new_idx

        return sorted_states

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        计算每个时间点各状态的后验概率。

        使用前向-后向算法计算后验概率 P(z_t | X)，
        表示在给定所有观测下，每个时刻处于各状态的概率。

        参数:
            X: 特征矩阵

        返回:
            后验概率矩阵，shape=(n_samples, n_states)
        """
        if self.model is None:
            raise RuntimeError("模型未训练，请先调用fit()")

        # 前向-后向算法计算后验概率
        raw_proba = self.model.predict_proba(X)

        # 按排序后的状态顺序重排列
        sorted_proba = np.zeros_like(raw_proba)
        for new_idx, old_idx in enumerate(self.state_order):
            sorted_proba[:, new_idx] = raw_proba[:, old_idx]

        return sorted_proba

    def get_transition_matrix(self) -> np.ndarray:
        """
        获取状态转移概率矩阵（已排序）。

        转移矩阵A[i,j]表示从状态i转移到状态j的概率。
        对角线元素A[i,i]表示停留在同一状态的概率，
        1 - A[i,i]表示离开该状态的概率。

        返回:
            转移概率矩阵，shape=(n_states, n_states)
        """
        if self.model is None:
            raise RuntimeError("模型未训练，请先调用fit()")

        raw_matrix = self.model.transmat_

        # 按排序后的状态顺序重排行列
        sorted_matrix = np.zeros_like(raw_matrix)
        for new_i, old_i in enumerate(self.state_order):
            for new_j, old_j in enumerate(self.state_order):
                sorted_matrix[new_i, new_j] = raw_matrix[old_i, old_j]

        return sorted_matrix

    def get_state_statistics(self, X: np.ndarray,
                             feature_names: Optional[List[str]] = None
                             ) -> pd.DataFrame:
        """
        获取各状态的统计特征。

        包括：
            - 各特征均值
            - 各特征方差
            - 预期持续时间（基于转移矩阵）
            - 状态出现频率

        参数:
            X: 特征矩阵
            feature_names: 特征名列表

        返回:
            状态统计DataFrame
        """
        if self.model is None:
            raise RuntimeError("模型未训练，请先调用fit()")

        if feature_names is None:
            feature_names = self.feature_names if self.feature_names else \
                [f"feature_{i}" for i in range(X.shape[1])]

        states = self.predict(X)
        transmat = self.get_transition_matrix()

        stats_list = []
        for i in range(self.n_states):
            # 获取该状态对应的原始模型索引
            old_idx = self.state_order[i]
            state_mask = states == i
            state_data = X[state_mask]

            # 计算状态统计
            stat = {
                "状态": self.state_labels[i],
                "状态编号": i,
                "样本数": int(state_mask.sum()),
                "出现频率": f"{state_mask.mean() * 100:.1f}%",
                # 预期持续时间 = 1 / (1 - A[i,i])
                "预期持续(期)": f"{1.0 / (1.0 - transmat[i, i]):.1f}",
            }

            # 各特征均值
            if len(state_data) > 0:
                for j, fname in enumerate(feature_names):
                    stat[f"{fname}_均值"] = float(state_data[:, j].mean())
                    stat[f"{fname}_方差"] = float(state_data[:, j].var())
            else:
                for j, fname in enumerate(feature_names):
                    stat[f"{fname}_均值"] = 0.0
                    stat[f"{fname}_方差"] = 0.0

            stats_list.append(stat)

        return pd.DataFrame(stats_list)

    def get_state_duration_analysis(self) -> pd.DataFrame:
        """
        分析各状态的持续时间特征。

        基于转移矩阵计算：
            - 预期持续时间: E[T] = 1 / (1 - p_stay)
            - 半衰期: t_1/2 = ln(0.5) / ln(p_stay)
            - 留存概率: p_stay = A[i,i]

        返回:
            持续时间分析DataFrame
        """
        if self.model is None:
            raise RuntimeError("模型未训练，请先调用fit()")

        transmat = self.get_transition_matrix()
        results = []

        for i in range(self.n_states):
            p_stay = transmat[i, i]
            expected_duration = 1.0 / (1.0 - p_stay) if p_stay < 1.0 else float("inf")
            half_life = np.log(0.5) / np.log(p_stay) if p_stay > 0 and p_stay < 1.0 else float("inf")

            results.append({
                "状态": self.state_labels[i],
                "留存概率": f"{p_stay:.4f}",
                "离开概率": f"{1 - p_stay:.4f}",
                "预期持续时间(期)": f"{expected_duration:.1f}",
                "半衰期(期)": f"{half_life:.1f}" if np.isfinite(half_life) else "∞",
            })

        return pd.DataFrame(results)

    def get_current_regime(self, X: np.ndarray) -> Dict:
        """
        获取当前市场状态分析。

        参数:
            X: 特征矩阵（最后一行为最新数据）

        返回:
            包含当前状态、概率分布、趋势预测的字典
        """
        if self.model is None:
            raise RuntimeError("模型未训练，请先调用fit()")

        proba = self.predict_proba(X)
        current_proba = proba[-1]  # 最后一行的概率分布
        current_state = np.argmax(current_proba)

        # 预测下一状态（基于转移矩阵）
        transmat = self.get_transition_matrix()
        next_proba = transmat[current_state, :]

        return {
            "当前状态": self.state_labels[current_state],
            "当前状态编号": int(current_state),
            "状态概率分布": {
                self.state_labels[i]: float(current_proba[i])
                for i in range(self.n_states)
            },
            "下一期状态预测": {
                self.state_labels[i]: float(next_proba[i])
                for i in range(self.n_states)
            },
            "状态置信度": f"{current_proba[current_state] * 100:.1f}%",
        }

    def score(self, X: np.ndarray) -> float:
        """
        计算观测序列的对数似然（模型评分）。

        参数:
            X: 特征矩阵

        返回:
            对数似然值
        """
        if self.model is None:
            raise RuntimeError("模型未训练，请先调用fit()")
        return float(self.model.score(X))

    def get_model_summary(self) -> Dict:
        """
        返回模型摘要信息。

        返回:
            包含模型参数和性能的字典
        """
        if self.model is None:
            raise RuntimeError("模型未训练，请先调用fit()")

        return {
            "模型类型": "GaussianHMM",
            "状态数": self.n_states,
            "协方差类型": "full",
            "最大迭代次数": self.n_iter,
            "收敛标志": bool(self.monitor_.converged) if hasattr(self, 'monitor_') else "N/A",
            "状态标签": self.state_labels,
            "随机种子": self.random_state,
        }
