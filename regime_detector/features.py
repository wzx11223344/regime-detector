# -*- coding: utf-8 -*-
"""
特征工程模块 (features.py)
==========================
构建用于HMM市场状态检测的特征矩阵。
包括：对数收益率、滚动波动率、成交量变化率、换手率、动量指标、RSI等。

所有特征均基于真实市场数据计算，不引入任何随机或伪造数据。
"""

import pandas as pd
import numpy as np
from typing import List, Optional
from sklearn.preprocessing import StandardScaler


class FeatureEngineer:
    """特征工程器，将原始OHLCV数据转换为HMM可用的特征矩阵。"""

    def __init__(self, rsi_period: int = 14, mom_period: int = 10,
                 vol_window: int = 20):
        """
        初始化特征工程器。

        参数:
            rsi_period: RSI计算周期，默认14日
            mom_period: 动量指标计算周期，默认10日
            vol_window: 滚动波动率窗口，默认20日
        """
        self.rsi_period = rsi_period
        self.mom_period = mom_period
        self.vol_window = vol_window
        self.scaler = StandardScaler()
        self.feature_names: List[str] = []

    def compute_log_return(self, df: pd.DataFrame) -> pd.Series:
        """
        计算对数收益率: r_t = ln(P_t / P_{t-1})

        对数收益率具有良好的可加性，适合HMM建模。
        """
        return np.log(df["close"] / df["close"].shift(1))

    def compute_rolling_volatility(self, df: pd.DataFrame,
                                   window: Optional[int] = None) -> pd.Series:
        """
        计算滚动波动率（GARCH式特征）。

        使用对数收益率的滚动标准差作为波动率代理。
        同时计算滚动方差（GARCH中的条件方差代理）。

        参数:
            window: 滚动窗口大小，默认使用self.vol_window
        """
        if window is None:
            window = self.vol_window

        log_ret = np.log(df["close"] / df["close"].shift(1))
        return log_ret.rolling(window=window).std()

    def compute_vol_squared_returns(self, df: pd.DataFrame) -> pd.Series:
        """
        计算平方收益率，作为GARCH模型中条件方差的直接代理。

        在GARCH(1,1)中，条件方差与滞后平方收益率相关，
        因此平方收益率是重要的波动率特征。
        """
        log_ret = np.log(df["close"] / df["close"].shift(1))
        return log_ret ** 2

    def compute_volume_change_rate(self, df: pd.DataFrame) -> pd.Series:
        """
        计算成交量变化率: VR_t = (V_t - V_{t-1}) / V_{t-1}

        成交量的突变往往预示着市场状态的切换。
        """
        vol = df["volume"].astype(float)
        return vol.pct_change()

    def compute_turnover_indicator(self, df: pd.DataFrame,
                                   window: int = 20) -> pd.Series:
        """
        计算换手率指标（成交额相对滚动均值的偏离）。

        如果数据中包含turnover_rate列，直接使用；
        否则用成交额的滚动Z-score作为代理指标。

        参数:
            window: 滚动窗口大小
        """
        if "turnover_rate" in df.columns:
            # 直接使用数据中的换手率
            tr = df["turnover_rate"].astype(float)
            # 计算换手率的滚动Z-score，衡量相对活跃度
            rolling_mean = tr.rolling(window=window).mean()
            rolling_std = tr.rolling(window=window).std()
            z_score = (tr - rolling_mean) / rolling_std
            return z_score
        else:
            # 使用成交额作为代理
            if "turnover" in df.columns:
                turnover = df["turnover"].astype(float)
            else:
                # 用成交量*收盘价估算成交额
                turnover = df["volume"].astype(float) * df["close"].astype(float)

            rolling_mean = turnover.rolling(window=window).mean()
            rolling_std = turnover.rolling(window=window).std()
            z_score = (turnover - rolling_mean) / rolling_std
            return z_score

    def compute_momentum(self, df: pd.DataFrame,
                         period: Optional[int] = None) -> pd.Series:
        """
        计算动量指标(MOM): MOM_t = P_t - P_{t-n}

        动量为正表示上涨趋势，为负表示下跌趋势。
        动量的绝对值大小反映趋势强度。

        参数:
            period: 动量周期，默认使用self.mom_period
        """
        if period is None:
            period = self.mom_period
        return df["close"] - df["close"].shift(period)

    def compute_rsi(self, df: pd.DataFrame,
                    period: Optional[int] = None) -> pd.Series:
        """
        计算RSI相对强弱指标。

        RSI = 100 - 100 / (1 + RS)
        RS = N日内平均涨幅 / N日内平均跌幅

        RSI > 70: 超买区域
        RSI < 30: 超卖区域
        30-70: 正常区域

        使用Wilder平滑法计算平均涨跌幅。

        参数:
            period: RSI周期，默认使用self.rsi_period
        """
        if period is None:
            period = self.rsi_period

        delta = df["close"].diff()
        # 分离上涨和下跌
        gain = delta.where(delta > 0, 0.0)
        loss = (-delta).where(delta < 0, 0.0)

        # 使用Wilder平滑法（指数移动平均）
        # 第一个平均值用简单平均
        avg_gain = gain.rolling(window=period, min_periods=period).mean()
        avg_loss = loss.rolling(window=period, min_periods=period).mean()

        # Wilder平滑: avg = (prev_avg * (n-1) + current) / n
        for i in range(period, len(df)):
            if pd.notna(avg_gain.iloc[i - 1]) and pd.notna(avg_loss.iloc[i - 1]):
                avg_gain.iloc[i] = (avg_gain.iloc[i - 1] * (period - 1) +
                                    gain.iloc[i]) / period
                avg_loss.iloc[i] = (avg_loss.iloc[i - 1] * (period - 1) +
                                    loss.iloc[i]) / period

        # 计算RS和RSI
        rs = avg_gain / avg_loss.replace(0, np.nan)
        rsi = 100 - (100 / (1 + rs))

        # 处理除零情况：当avg_loss为0时，RSI=100
        rsi = rsi.fillna(100.0)

        return rsi

    def compute_price_acceleration(self, df: pd.DataFrame) -> pd.Series:
        """
        计算价格加速度（收益率的差分）。

        加速度反映趋势的变化速度，是二阶导数特征。
        正加速度表示趋势在加速，负加速度表示趋势在减速。
        """
        log_ret = np.log(df["close"] / df["close"].shift(1))
        return log_ret.diff()

    def compute_features(self, df: pd.DataFrame,
                         feature_list: Optional[List[str]] = None
                         ) -> pd.DataFrame:
        """
        计算全部特征并返回特征DataFrame。

        参数:
            df: 原始OHLCV数据
            feature_list: 指定使用的特征列表，None表示使用全部

        返回:
            包含所有特征的DataFrame
        """
        result = df.copy()

        # 对数收益率
        result["log_return"] = self.compute_log_return(df)

        # 滚动波动率
        result["rolling_vol"] = self.compute_rolling_volatility(df)

        # 平方收益率（GARCH条件方差代理）
        result["squared_return"] = self.compute_vol_squared_returns(df)

        # 成交量变化率
        result["volume_change"] = self.compute_volume_change_rate(df)

        # 换手率指标
        result["turnover_indicator"] = self.compute_turnover_indicator(df)

        # 动量指标
        result["momentum"] = self.compute_momentum(df)

        # RSI指标
        result["rsi"] = self.compute_rsi(df)

        # 价格加速度
        result["price_accel"] = self.compute_price_acceleration(df)

        # 选择指定特征
        all_features = [
            "log_return", "rolling_vol", "squared_return",
            "volume_change", "turnover_indicator",
            "momentum", "rsi", "price_accel",
        ]

        if feature_list is not None:
            # 验证特征名有效
            invalid = set(feature_list) - set(all_features)
            if invalid:
                raise ValueError(f"无效的特征名: {invalid}。可用特征: {all_features}")
            use_features = feature_list
        else:
            use_features = all_features

        self.feature_names = use_features
        return result

    def build_feature_matrix(self, df: pd.DataFrame,
                             feature_list: Optional[List[str]] = None,
                             scale: bool = True) -> tuple:
        """
        构建HMM可用的特征矩阵。

        参数:
            df: 包含特征的DataFrame（需先调用compute_features）
            feature_list: 使用的特征列表
            scale: 是否标准化（HMM通常需要标准化）

        返回:
            (feature_matrix, feature_names, dates)
            - feature_matrix: numpy数组，shape=(n_samples, n_features)
            - feature_names: 特征名列表
            - dates: 对应的日期列表
        """
        if feature_list is None:
            feature_list = self.feature_names if self.feature_names else [
                "log_return", "rolling_vol", "squared_return",
                "volume_change", "turnover_indicator",
                "momentum", "rsi", "price_accel",
            ]

        # 提取特征列，去除NaN行
        feature_df = df[["date"] + feature_list].dropna()

        if len(feature_df) < 50:
            raise ValueError(
                f"有效数据不足: 去除NaN后仅剩{len(feature_df)}行，"
                f"至少需要50行数据用于HMM训练"
            )

        dates = feature_df["date"].values
        matrix = feature_df[feature_list].values

        # 标准化处理
        if scale:
            matrix = self.scaler.fit_transform(matrix)

        return matrix, feature_list, dates

    def get_feature_description(self) -> dict:
        """
        返回各特征的描述说明。

        用于报告生成和文档展示。
        """
        return {
            "log_return": "对数收益率 ln(P_t/P_{t-1})，衡量价格变动",
            "rolling_vol": "20日滚动波动率，衡量波动水平",
            "squared_return": "平方收益率，GARCH条件方差代理",
            "volume_change": "成交量变化率，衡量交易活跃度变化",
            "turnover_indicator": "换手率Z-score，衡量相对活跃度",
            "momentum": "动量指标 P_t - P_{t-n}，衡量趋势方向",
            "rsi": "RSI相对强弱指标，衡量超买超卖",
            "price_accel": "价格加速度，收益率差分，衡量趋势变化速度",
        }
