# -*- coding: utf-8 -*-
"""
数据获取模块 (data.py)
======================
使用akshare获取真实的A股指数和个股历史数据。
支持多频率（日/周/月）数据，计算对数收益率和滚动波动率。

所有数据均为真实市场数据，禁止任何随机或伪造数据。
"""

import akshare as ak
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Optional, Union


class DataFetcher:
    """真实市场数据获取器，基于akshare接口。"""

    # 常见指数代码映射（akshare格式）
    INDEX_MAP = {
        "sh000001": "上证指数",
        "sh000300": "沪深300",
        "sh000016": "上证50",
        "sh000905": "中证500",
        "sz399001": "深证成指",
        "sz399006": "创业板指",
        "sh000688": "科创50",
    }

    def __init__(self):
        """初始化数据获取器。"""
        self._cache = {}  # 简单内存缓存，避免重复请求

    def fetch_index_data(
        self,
        symbol: str = "sh000001",
        start_date: str = "20200101",
        end_date: Optional[str] = None,
    ) -> pd.DataFrame:
        """
        获取指数历史日线数据。

        参数:
            symbol: 指数代码，如 "sh000001"（上证指数）、"sh000300"（沪深300）
            start_date: 开始日期，格式 "YYYYMMDD"
            end_date: 结束日期，格式 "YYYYMMDD"，默认为今天

        返回:
            DataFrame，包含 date, open, high, low, close, volume 列
        """
        if end_date is None:
            end_date = datetime.now().strftime("%Y%m%d")

        cache_key = f"index_{symbol}_{start_date}_{end_date}"
        if cache_key in self._cache:
            return self._cache[cache_key].copy()

        try:
            # 使用akshare获取指数日线数据
            # akshare的 stock_zh_index_daily 接口返回历史日线
            df = ak.stock_zh_index_daily(symbol=symbol)
        except Exception as e:
            # 尝试备用接口 stock_zh_index_daily_em
            try:
                df = ak.stock_zh_index_daily_em(symbol=symbol)
            except Exception as e2:
                raise RuntimeError(
                    f"获取指数数据失败: {symbol}。"
                    f"主接口错误: {e}；备用接口错误: {e2}"
                )

        # 统一列名处理
        df = self._normalize_columns(df)

        # 日期过滤
        df["date"] = pd.to_datetime(df["date"])
        mask = (df["date"] >= pd.to_datetime(start_date, format="%Y%m%d")) & \
               (df["date"] <= pd.to_datetime(end_date, format="%Y%m%d"))
        df = df.loc[mask].sort_values("date").reset_index(drop=True)

        if df.empty:
            raise ValueError(
                f"过滤后数据为空，请检查日期范围: {start_date} ~ {end_date}，"
                f"指数代码: {symbol}"
            )

        # 计算对数收益率
        df["log_return"] = np.log(df["close"] / df["close"].shift(1))

        # 计算滚动波动率（20日窗口）
        df["rolling_vol_20"] = df["log_return"].rolling(window=20).std()

        # 缓存并返回
        self._cache[cache_key] = df.copy()
        return df

    def fetch_stock_data(
        self,
        symbol: str = "000001",
        start_date: str = "20200101",
        end_date: Optional[str] = None,
        adjust: str = "qfq",
    ) -> pd.DataFrame:
        """
        获取个股历史数据（前复权）。

        参数:
            symbol: 股票代码，如 "000001"（平安银行）
            start_date: 开始日期，格式 "YYYYMMDD"
            end_date: 结束日期，格式 "YYYYMMDD"
            adjust: 复权类型，"qfq"前复权, "hfq"后复权, ""不复权

        返回:
            DataFrame，包含 date, open, high, low, close, volume, turnover 列
        """
        if end_date is None:
            end_date = datetime.now().strftime("%Y%m%d")

        cache_key = f"stock_{symbol}_{start_date}_{end_date}_{adjust}"
        if cache_key in self._cache:
            return self._cache[cache_key].copy()

        try:
            # 使用akshare获取个股历史数据
            # period可选: daily/weekly/monthly
            df = ak.stock_zh_a_hist(
                symbol=symbol,
                period="daily",
                start_date=start_date,
                end_date=end_date,
                adjust=adjust,
            )
        except Exception as e:
            raise RuntimeError(f"获取个股数据失败: {symbol}。错误: {e}")

        # 统一列名处理
        df = self._normalize_columns(df)

        # 计算对数收益率
        df["log_return"] = np.log(df["close"] / df["close"].shift(1))

        # 计算滚动波动率（20日窗口）
        df["rolling_vol_20"] = df["log_return"].rolling(window=20).std()

        self._cache[cache_key] = df.copy()
        return df

    def fetch_data(
        self,
        ticker: Optional[str] = None,
        index: Optional[str] = None,
        start_date: str = "20200101",
        end_date: Optional[str] = None,
        freq: str = "daily",
    ) -> pd.DataFrame:
        """
        统一数据获取入口，根据参数自动选择指数或个股。

        参数:
            ticker: 个股代码（如 "000001"），与index二选一
            index: 指数代码（如 "sh000300"），与ticker二选一
            start_date: 开始日期
            end_date: 结束日期
            freq: 数据频率，daily/weekly/monthly

        返回:
            DataFrame，统一格式的历史行情数据
        """
        if index is not None:
            df = self.fetch_index_data(index, start_date, end_date)
            df["symbol"] = index
            df["name"] = self.INDEX_MAP.get(index, index)
        elif ticker is not None:
            df = self.fetch_stock_data(ticker, start_date, end_date)
            df["symbol"] = ticker
            df["name"] = ticker
        else:
            raise ValueError("必须指定 ticker 或 index 参数")

        # 多频率重采样
        if freq != "daily":
            df = self._resample_frequency(df, freq)

        return df

    def _resample_frequency(self, df: pd.DataFrame, freq: str) -> pd.DataFrame:
        """
        将日线数据重采样为周线或月线。

        参数:
            df: 日线DataFrame
            freq: 目标频率，"weekly" 或 "monthly"

        返回:
            重采样后的DataFrame
        """
        df = df.set_index("date")

        if freq == "weekly":
            # W-FRI 表示以周五为周末
            resampled = df.resample("W-FRI").agg({
                "open": "first",
                "high": "max",
                "low": "min",
                "close": "last",
                "volume": "sum",
            }).dropna()
        elif freq == "monthly":
            resampled = df.resample("M").agg({
                "open": "first",
                "high": "max",
                "low": "min",
                "close": "last",
                "volume": "sum",
            }).dropna()
        else:
            raise ValueError(f"不支持的频率: {freq}，可选: daily/weekly/monthly")

        resampled = resampled.reset_index()

        # 重新计算对数收益率和滚动波动率
        resampled["log_return"] = np.log(
            resampled["close"] / resampled["close"].shift(1)
        )
        resampled["rolling_vol_20"] = resampled["log_return"].rolling(
            window=min(20, len(resampled) // 2)
        ).std()

        # 保留symbol和name列
        if "symbol" in df.columns:
            resampled["symbol"] = df["symbol"].iloc[0]
        if "name" in df.columns:
            resampled["name"] = df["name"].iloc[0]

        return resampled

    def _normalize_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        统一不同akshare接口返回的列名格式。

        akshare不同接口返回的列名可能为中文或英文，
        此方法将其统一为英文标准列名。
        """
        # 中文列名到英文的映射
        column_map = {
            "日期": "date",
            "开盘": "open",
            "最高": "high",
            "最低": "low",
            "收盘": "close",
            "成交量": "volume",
            "成交额": "turnover",
            "振幅": "amplitude",
            "涨跌幅": "pct_change",
            "涨跌额": "change",
            "换手率": "turnover_rate",
        }

        # 重命名列
        df = df.rename(columns=column_map)

        # 确保必要的列存在
        required_cols = ["date", "open", "high", "low", "close", "volume"]
        for col in required_cols:
            if col not in df.columns:
                if col == "volume" and "vol" in df.columns:
                    df = df.rename(columns={"vol": "volume"})
                else:
                    raise ValueError(f"数据缺少必要列: {col}，当前列: {list(df.columns)}")

        # 只保留需要的列
        keep_cols = [c for c in required_cols if c in df.columns]
        extra_cols = [c for c in df.columns if c not in required_cols]
        df = df[keep_cols + extra_cols]

        return df

    def get_available_indices(self) -> dict:
        """返回支持的指数代码列表。"""
        return self.INDEX_MAP.copy()
