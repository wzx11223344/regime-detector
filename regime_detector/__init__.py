# -*- coding: utf-8 -*-
"""
市场状态智能检测器 (Regime Detector)
=====================================
基于隐马尔可夫模型(HMM)和变点检测算法的金融市场状态识别工具。

核心功能：
    - 数据获取（akshare真实数据）
    - 特征工程（收益率、波动率、动量、RSI等）
    - HMM市场状态检测（牛市/熊市/震荡/危机）
    - 变点检测（CUSUM/PELT/贝叶斯在线）
    - 波动率建模（GARCH(1,1)）
    - HTML可视化报告生成

作者: Regime Detector Team
许可证: MIT
"""

__version__ = "1.0.0"
__author__ = "Regime Detector Team"
__license__ = "MIT"

# 导入核心模块，方便上层调用
from .data import DataFetcher
from .features import FeatureEngineer
from .hmm_model import HMMRegimeDetector
from .changepoint import ChangepointDetector
from .volatility import VolatilityAnalyzer
from .report import ReportGenerator

__all__ = [
    "DataFetcher",
    "FeatureEngineer",
    "HMMRegimeDetector",
    "ChangepointDetector",
    "VolatilityAnalyzer",
    "ReportGenerator",
]
