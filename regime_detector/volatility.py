# -*- coding: utf-8 -*-
"""
波动率分析模块 (volatility.py)
================================
实现GARCH(1,1)波动率建模、波动率状态分类和聚集效应分析。

功能:
    1. GARCH(1,1)条件波动率建模（arch库）
    2. 波动率状态分类（低波/中波/高波）
    3. 波动率聚集效应分析
    4. 实现波动率与GARCH预测波动率对比

所有计算基于真实市场数据。
"""

import numpy as np
import pandas as pd
from arch import arch_model
from scipy import stats
from statsmodels.stats.diagnostic import acorr_ljungbox
from typing import Dict, List, Optional, Tuple


class VolatilityAnalyzer:
    """波动率分析器，基于GARCH模型和统计分析。"""

    # 波动率状态标签
    VOL_STATES = ["低波动", "中波动", "高波动"]

    # 波动率状态颜色
    VOL_COLORS = {
        "低波动": "#2ecc71",
        "中波动": "#f39c12",
        "高波动": "#e74c3c",
    }

    def __init__(self, window: int = 20):
        """
        初始化波动率分析器。

        参数:
            window: 滚动窗口大小（用于实现波动率计算）
        """
        self.window = window
        self.garch_result = None
        self.conditional_vol = None

    def compute_realized_volatility(self, returns: np.ndarray,
                                    window: Optional[int] = None) -> np.ndarray:
        """
        计算实现波动率（历史波动率）。

        实现波动率使用滚动窗口的标准差计算，
        是对过去波动水平的回顾性度量。

        公式: RV_t = std(r_{t-w+1:t}) * sqrt(252)
        （年化处理，假设252个交易日）

        参数:
            returns: 收益率序列
            window: 滚动窗口

        返回:
            实现波动率序列（年化）
        """
        if window is None:
            window = self.window

        returns = pd.Series(returns)
        # 滚动标准差并年化
        realized_vol = returns.rolling(window=window).std() * np.sqrt(252)
        return realized_vol.values

    def fit_garch(self, returns: np.ndarray,
                  p: int = 1, q: int = 1) -> "VolatilityAnalyzer":
        """
        拟合GARCH(p,q)模型。

        GARCH(1,1)模型:
            sigma^2_t = omega + alpha * r^2_{t-1} + beta * sigma^2_{t-1}

        其中:
            - omega: 长期方差项
            - alpha: ARCH项系数（近期冲击影响）
            - beta: GARCH项系数（前期方差影响）
            - alpha + beta < 1: 平稳性条件

        参数:
            returns: 收益率序列（百分比或小数均可）
            p: GARCH阶数（滞后sigma^2项数）
            q: ARCH阶数（滞后r^2项数）

        返回:
            self（支持链式调用）
        """
        returns = np.asarray(returns, dtype=float)
        # 去除NaN
        returns = returns[np.isfinite(returns)]

        if len(returns) < 50:
            raise ValueError(f"数据不足: GARCH拟合至少需要50个有效值，当前{len(returns)}")

        # 缩放收益率（arch库建议使用百分比收益率）
        # 如果收益率太小（如0.01级别），乘以100
        if np.abs(returns).mean() < 0.1:
            scaled_returns = returns * 100
        else:
            scaled_returns = returns

        try:
            # 创建GARCH模型
            # vol='Garch'指定GARCH模型
            # mean='Zero'假设均值为零（或使用'Constant'）
            # dist='normal'假设正态分布
            model = arch_model(
                scaled_returns,
                vol="Garch",
                p=p,
                q=q,
                mean="Constant",
                dist="normal",
            )

            # 拟合模型
            self.garch_result = model.fit(disp="off", show_warning=False)
            # 条件波动率（缩放回原始尺度）
            self.conditional_vol = self.garch_result.conditional_volatility / 100

        except Exception as e:
            raise RuntimeError(f"GARCH模型拟合失败: {e}")

        return self

    def get_garch_parameters(self) -> Dict:
        """
        获取GARCH模型参数。

        返回:
            包含模型参数和统计量的字典
        """
        if self.garch_result is None:
            raise RuntimeError("模型未拟合，请先调用fit_garch()")

        params = self.garch_result.params
        return {
            "mu（均值）": float(params.get("mu", 0)),
            "omega（长期方差）": float(params.get("omega", 0)),
            "alpha[1]（ARCH系数）": float(params.get("alpha[1]", 0)),
            "beta[1]（GARCH系数）": float(params.get("beta[1]", 0)),
            "alpha+beta（持续性）": float(
                params.get("alpha[1]", 0) + params.get("beta[1]", 0)
            ),
            "对数似然": float(self.garch_result.loglikelihood),
            "AIC": float(self.garch_result.aic),
            "BIC": float(self.garch_result.bic),
            "是否平稳": bool(
                params.get("alpha[1]", 0) + params.get("beta[1]", 0) < 1.0
            ),
        }

    def forecast_volatility(self, horizon: int = 5) -> np.ndarray:
        """
        预测未来波动率。

        使用GARCH模型进行向前h步预测。

        参数:
            horizon: 预测步数

        返回:
            预测波动率数组
        """
        if self.garch_result is None:
            raise RuntimeError("模型未拟合，请先调用fit_garch()")

        # 向前预测
        forecast = self.garch_result.forecast(horizon=horizon)
        # 预测方差（缩放回原始尺度）
        forecast_var = forecast.variance.values[-1, :] / 10000
        forecast_vol = np.sqrt(forecast_var)
        return forecast_vol

    def classify_volatility_states(self, returns: np.ndarray,
                                   method: str = "quantile"
                                   ) -> Tuple[np.ndarray, Dict]:
        """
        将波动率分为低/中/高三个状态。

        分类方法:
            - quantile: 基于分位数（33%/67%）
            - fixed: 基于固定阈值（年化15%/30%）
            - garch: 基于GARCH条件波动率分位数

        参数:
            returns: 收益率序列
            method: 分类方法

        返回:
            (state_labels, thresholds)
            - state_labels: 0=低波动, 1=中波动, 2=高波动
            - thresholds: 各状态阈值
        """
        returns = np.asarray(returns, dtype=float)

        if method == "garch" and self.conditional_vol is not None:
            # 使用GARCH条件波动率分类
            vol = self.conditional_vol
            q1 = np.nanpercentile(vol, 33)
            q2 = np.nanpercentile(vol, 67)
        elif method == "fixed":
            # 固定阈值（年化）
            vol = self.compute_realized_volatility(returns)
            q1 = 0.15  # 15%年化
            q2 = 0.30  # 30%年化
        else:
            # 默认分位数法
            vol = self.compute_realized_volatility(returns)
            q1 = np.nanpercentile(vol, 33)
            q2 = np.nanpercentile(vol, 67)

        # 分类
        states = np.ones(len(returns), dtype=int)  # 默认中波动
        valid_mask = np.isfinite(vol)

        states[valid_mask & (vol < q1)] = 0  # 低波动
        states[valid_mask & (vol >= q2)] = 2  # 高波动
        # NaN处保持中波动

        thresholds = {
            "低波动上限": float(q1),
            "高波动下限": float(q2),
            "分类方法": method,
        }

        return states, thresholds

    def analyze_volatility_clustering(self, returns: np.ndarray) -> Dict:
        """
        分析波动率聚集效应。

        波动率聚集是指：高波动期后往往跟随高波动期，
        低波动期后往往跟随低波动期。

        分析方法:
            1. 平方收益率的自相关函数（ACF）
            2. Ljung-Box检验（检验自相关性是否显著）
            3. GARCH持续性参数（alpha+beta）

        参数:
            returns: 收益率序列

        返回:
            聚集效应分析结果
        """
        returns = np.asarray(returns, dtype=float)
        returns = returns[np.isfinite(returns)]
        squared_returns = returns ** 2

        # 计算平方收益率的自相关
        max_lag = min(20, len(squared_returns) // 4)
        acf_values = []
        for lag in range(1, max_lag + 1):
            acf_val = pd.Series(squared_returns).autocorr(lag=lag)
            acf_values.append(float(acf_val) if np.isfinite(acf_val) else 0.0)

        # Ljung-Box检验
        # 原假设：数据是独立同分布的（无自相关）
        # p值<0.05表示存在显著的自相关（即存在波动率聚集）
        # 使用statsmodels的acorr_ljungbox（scipy新版本已移除此函数）
        lb_result = acorr_ljungbox(squared_returns, lags=[max_lag])
        lb_stat = lb_result["lb_stat"].values
        lb_pvalue = lb_result["lb_pvalue"].values

        # GARCH持续性
        persistence = None
        if self.garch_result is not None:
            params = self.garch_result.params
            persistence = float(
                params.get("alpha[1]", 0) + params.get("beta[1]", 0)
            )

        return {
            "平方收益率ACF": acf_values,
            "ACF最大滞后": max_lag,
            "Ljung-Box统计量": float(lb_stat[0]),
            "Ljung-Box p值": float(lb_pvalue[0]),
            "存在波动率聚集": bool(lb_pvalue[0] < 0.05),
            "GARCH持续性(alpha+beta)": persistence,
            "聚集强度": "强" if lb_pvalue[0] < 0.01 else
                       ("中" if lb_pvalue[0] < 0.05 else "弱"),
        }

    def compare_realized_vs_forecast(self, returns: np.ndarray) -> Dict:
        """
        对比实现波动率与GARCH预测波动率。

        实现波动率是回顾性的（基于历史数据），
        GARCH条件波动率是前瞻性的（基于模型预测）。

        两者差异反映了市场预期与实际波动的偏离。

        参数:
            returns: 收益率序列

        返回:
            对比分析结果
        """
        realized_vol = self.compute_realized_volatility(returns)

        if self.conditional_vol is None:
            # 如果未拟合GARCH，只返回实现波动率
            return {
                "实现波动率均值": float(np.nanmean(realized_vol)),
                "实现波动率最大值": float(np.nanmax(realized_vol)),
                "实现波动率最小值": float(np.nanmin(realized_vol)),
                "GARCH条件波动率": "未拟合GARCH模型",
            }

        garch_vol = self.conditional_vol

        # 对齐长度
        min_len = min(len(realized_vol), len(garch_vol))
        realized = realized_vol[-min_len:]
        garch = garch_vol[-min_len:]

        # 计算差异
        valid_mask = np.isfinite(realized) & np.isfinite(garch)
        diff = realized[valid_mask] - garch[valid_mask]

        return {
            "实现波动率均值": float(np.nanmean(realized_vol)),
            "GARCH波动率均值": float(np.nanmean(garch_vol)),
            "波动率差异均值": float(np.mean(diff)),
            "波动率差异标准差": float(np.std(diff)),
            "实现>预测占比": f"{np.mean(diff > 0) * 100:.1f}%",
            "相关性": float(np.corrcoef(realized[valid_mask], garch[valid_mask])[0, 1]),
            "实现波动率序列": realized_vol.tolist(),
            "GARCH波动率序列": garch_vol.tolist(),
        }

    def get_volatility_summary(self, returns: np.ndarray) -> Dict:
        """
        获取波动率分析综合摘要。

        参数:
            returns: 收益率序列

        返回:
            综合摘要字典
        """
        summary = {}

        # 基本统计
        returns_clean = np.asarray(returns, dtype=float)
        returns_clean = returns_clean[np.isfinite(returns_clean)]
        summary["基本统计"] = {
            "日均收益率": float(returns_clean.mean()),
            "日波动率": float(returns_clean.std()),
            "年化波动率": float(returns_clean.std() * np.sqrt(252)),
            "偏度": float(stats.skew(returns_clean)),
            "峰度": float(stats.kurtosis(returns_clean)),
            "最大单日涨幅": float(returns_clean.max()),
            "最大单日跌幅": float(returns_clean.min()),
        }

        # GARCH参数
        if self.garch_result is not None:
            summary["GARCH参数"] = self.get_garch_parameters()

        # 波动率状态
        states, thresholds = self.classify_volatility_states(returns)
        state_counts = pd.Series(states).value_counts().sort_index()
        summary["波动率状态"] = {
            "低波动占比": f"{state_counts.get(0, 0) / len(states) * 100:.1f}%",
            "中波动占比": f"{state_counts.get(1, 0) / len(states) * 100:.1f}%",
            "高波动占比": f"{state_counts.get(2, 0) / len(states) * 100:.1f}%",
            "阈值": thresholds,
        }

        # 聚集效应
        summary["聚集效应"] = self.analyze_volatility_clustering(returns)

        # 实现vs预测
        summary["波动率对比"] = self.compare_realized_vs_forecast(returns)

        return summary
