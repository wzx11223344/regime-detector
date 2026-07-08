# -*- coding: utf-8 -*-
"""
报告生成模块 (report.py)
========================
使用matplotlib生成可视化图表，并构建HTML报告。
图表以base64编码嵌入HTML，实现单文件自包含报告。

生成图表:
    1. 价格走势+HMM状态着色图
    2. 状态转移概率热力图
    3. 变点检测标注图
    4. 波动率状态图
"""

import base64
import io
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # 非交互式后端，避免显示窗口
import matplotlib.pyplot as plt
from matplotlib.dates import DateFormatter
from typing import Dict, List, Optional, Any
from datetime import datetime


class ReportGenerator:
    """HTML报告生成器，含可视化图表。"""

    # 配色方案
    COLOR_SCHEME = {
        "牛市": "#2ecc71",
        "震荡": "#f39c12",
        "熊市": "#e74c3c",
        "危机": "#8b0000",
        "低波动": "#2ecc71",
        "中波动": "#f39c12",
        "高波动": "#e74c3c",
    }

    def __init__(self):
        """初始化报告生成器，配置中文字体。"""
        self._setup_font()

    def _setup_font(self):
        """配置matplotlib中文字体支持。"""
        # 尝试多种中文字体
        font_candidates = [
            "Microsoft YaHei",  # 微软雅黑（Windows）
            "SimHei",           # 黑体（Windows）
            "PingFang SC",      # 苹方（macOS）
            "WenQuanYi Micro Hei",  # 文泉驿（Linux）
            "Noto Sans CJK SC",     # 思源黑体（Linux）
        ]

        for font in font_candidates:
            try:
                from matplotlib.font_manager import FontProperties
                fp = FontProperties(family=font)
                if fp.get_name() != font:
                    continue
                plt.rcParams["font.sans-serif"] = [font, "DejaVu Sans"]
                plt.rcParams["axes.unicode_minus"] = False
                return
            except Exception:
                continue

        # 如果都找不到，设置默认并禁用unicode减号
        plt.rcParams["font.sans-serif"] = ["DejaVu Sans"]
        plt.rcParams["axes.unicode_minus"] = False

    def _fig_to_base64(self, fig: plt.Figure) -> str:
        """将matplotlib图形转换为base64编码字符串。"""
        buf = io.BytesIO()
        fig.savefig(buf, format="png", dpi=150, bbox_inches="tight",
                    facecolor="white", edgecolor="none")
        plt.close(fig)
        buf.seek(0)
        img_base64 = base64.b64encode(buf.read()).decode("utf-8")
        return f"data:image/png;base64,{img_base64}"

    def plot_price_with_regimes(self, df: pd.DataFrame,
                                states: np.ndarray,
                                state_labels: List[str]) -> str:
        """
        生成价格走势+HMM状态着色图。

        不同市场状态用不同背景颜色标注，直观展示状态切换。

        参数:
            df: 含date和close列的DataFrame
            states: HMM状态序列
            state_labels: 状态标签列表

        返回:
            base64编码的图片字符串
        """
        fig, ax = plt.subplots(figsize=(14, 6))

        dates = pd.to_datetime(df["date"])
        close = df["close"].values

        # 绘制价格曲线
        ax.plot(dates, close, color="#2c3e50", linewidth=1.2, label="收盘价")

        # 为每个状态段添加背景色
        unique_states = sorted(set(states))
        for state in unique_states:
            mask = states == state
            label = state_labels[state] if state < len(state_labels) else f"状态{state}"
            color = self.COLOR_SCHEME.get(label, "#bdc3c7")

            # 找到连续的状态段
            segments = self._find_contiguous_segments(mask)
            for start, end in segments:
                ax.axvspan(
                    dates.iloc[start], dates.iloc[min(end, len(dates) - 1)],
                    alpha=0.25, color=color, label=label if start == segments[0][0] else ""
                )

        ax.set_title("价格走势与市场状态标注", fontsize=14, fontweight="bold")
        ax.set_xlabel("日期", fontsize=11)
        ax.set_ylabel("价格", fontsize=11)
        ax.legend(loc="upper left", fontsize=9)
        ax.grid(True, alpha=0.3)

        # 格式化x轴日期
        ax.xaxis.set_major_formatter(DateFormatter("%Y-%m"))
        fig.autofmt_xdate()

        return self._fig_to_base64(fig)

    def plot_transition_heatmap(self, transmat: np.ndarray,
                                state_labels: List[str]) -> str:
        """
        生成状态转移概率热力图。

        参数:
            transmat: 转移概率矩阵
            state_labels: 状态标签列表

        返回:
            base64编码的图片字符串
        """
        fig, ax = plt.subplots(figsize=(7, 6))

        n = len(state_labels)
        im = ax.imshow(transmat, cmap="YlOrRd", aspect="auto", vmin=0, vmax=1)

        # 添加数值标注
        for i in range(n):
            for j in range(n):
                text_color = "white" if transmat[i, j] > 0.5 else "black"
                ax.text(j, i, f"{transmat[i, j]:.3f}",
                        ha="center", va="center",
                        color=text_color, fontsize=12, fontweight="bold")

        ax.set_xticks(range(n))
        ax.set_yticks(range(n))
        ax.set_xticklabels(state_labels, fontsize=11)
        ax.set_yticklabels(state_labels, fontsize=11)
        ax.set_title("状态转移概率矩阵", fontsize=14, fontweight="bold")
        ax.set_xlabel("目标状态", fontsize=11)
        ax.set_ylabel("当前状态", fontsize=11)

        # 添加颜色条
        cbar = fig.colorbar(im, ax=ax, shrink=0.8)
        cbar.set_label("转移概率", fontsize=10)

        return self._fig_to_base64(fig)

    def plot_changepoints(self, df: pd.DataFrame,
                          returns: np.ndarray,
                          changepoints: List[Dict]) -> str:
        """
        生成变点检测标注图。

        在收益率曲线上标注检测到的变点位置。

        参数:
            df: 含date列的DataFrame
            returns: 收益率序列
            changepoints: 变点列表

        返回:
            base64编码的图片字符串
        """
        fig, ax = plt.subplots(figsize=(14, 5))

        dates = pd.to_datetime(df["date"])
        # 取与returns等长的日期
        n = len(returns)
        dates = dates.iloc[-n:].reset_index(drop=True)

        # 绘制收益率
        ax.plot(dates, returns, color="#34495e", linewidth=0.8, alpha=0.7)
        ax.fill_between(dates, returns, 0, where=(returns >= 0),
                        color="#2ecc71", alpha=0.3, label="正收益")
        ax.fill_between(dates, returns, 0, where=(returns < 0),
                        color="#e74c3c", alpha=0.3, label="负收益")

        # 标注变点
        for cp in changepoints:
            pos = cp["位置"]
            if pos < n:
                algo = cp.get("算法", "未知")
                color = {"CUSUM": "#e74c3c", "PELT": "#9b59b6",
                         "Bayesian-Online": "#1abc9c"}.get(algo, "#e74c3c")
                ax.axvline(dates.iloc[pos], color=color, linewidth=1.5,
                           linestyle="--", alpha=0.8)
                ax.annotate(
                    f"{algo}\n(置信度:{cp.get('置信度', 0):.2f})",
                    xy=(dates.iloc[pos], returns[pos]),
                    fontsize=7, color=color,
                    xytext=(5, 15), textcoords="offset points",
                    arrowprops=dict(arrowstyle="->", color=color, lw=0.8),
                )

        ax.set_title("变点检测标注图", fontsize=14, fontweight="bold")
        ax.set_xlabel("日期", fontsize=11)
        ax.set_ylabel("对数收益率", fontsize=11)
        ax.legend(loc="upper right", fontsize=9)
        ax.grid(True, alpha=0.3)
        ax.xaxis.set_major_formatter(DateFormatter("%Y-%m"))
        fig.autofmt_xdate()

        return self._fig_to_base64(fig)

    def plot_volatility_states(self, df: pd.DataFrame,
                               returns: np.ndarray,
                               vol_states: np.ndarray,
                               realized_vol: np.ndarray,
                               garch_vol: Optional[np.ndarray] = None) -> str:
        """
        生成波动率状态图。

        显示实现波动率、GARCH条件波动率和波动率状态分类。

        参数:
            df: 含date列的DataFrame
            returns: 收益率序列
            vol_states: 波动率状态序列
            realized_vol: 实现波动率序列
            garch_vol: GARCH条件波动率（可选）

        返回:
            base64编码的图片字符串
        """
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 8),
                                       gridspec_kw={"height_ratios": [2, 1]})

        n = len(returns)
        dates = pd.to_datetime(df["date"]).iloc[-n:].reset_index(drop=True)

        # 上图：波动率曲线
        ax1.plot(dates, realized_vol, color="#2980b9", linewidth=1.2,
                 label="实现波动率", alpha=0.8)
        if garch_vol is not None:
            garch_aligned = garch_vol[-n:]
            ax1.plot(dates, garch_aligned, color="#e74c3c", linewidth=1.2,
                     label="GARCH条件波动率", alpha=0.8, linestyle="--")

        # 波动率状态背景色
        vol_labels = ["低波动", "中波动", "高波动"]
        for state in sorted(set(vol_states)):
            if state < 3:
                mask = vol_states == state
                label = vol_labels[state]
                color = self.COLOR_SCHEME.get(label, "#bdc3c7")
                segments = self._find_contiguous_segments(mask)
                for start, end in segments:
                    ax1.axvspan(
                        dates.iloc[start], dates.iloc[min(end, n - 1)],
                        alpha=0.15, color=color,
                        label=label if start == segments[0][0] else ""
                    )

        ax1.set_title("波动率分析", fontsize=14, fontweight="bold")
        ax1.set_ylabel("年化波动率", fontsize=11)
        ax1.legend(loc="upper left", fontsize=9)
        ax1.grid(True, alpha=0.3)
        ax1.xaxis.set_major_formatter(DateFormatter("%Y-%m"))

        # 下图：收益率
        ax2.bar(dates, returns, width=1, color=["#2ecc71" if r >= 0 else "#e74c3c"
                for r in returns], alpha=0.6)
        ax2.set_xlabel("日期", fontsize=11)
        ax2.set_ylabel("对数收益率", fontsize=11)
        ax2.grid(True, alpha=0.3)
        ax2.xaxis.set_major_formatter(DateFormatter("%Y-%m"))

        fig.autofmt_xdate()
        plt.tight_layout()

        return self._fig_to_base64(fig)

    def _find_contiguous_segments(self, mask: np.ndarray) -> List[tuple]:
        """找到布尔数组中连续为True的段。"""
        segments = []
        in_segment = False
        start = 0

        for i, val in enumerate(mask):
            if val and not in_segment:
                start = i
                in_segment = True
            elif not val and in_segment:
                segments.append((start, i - 1))
                in_segment = False

        if in_segment:
            segments.append((start, len(mask) - 1))

        return segments

    def _table_to_html(self, df: pd.DataFrame, title: str = "") -> str:
        """将DataFrame转换为HTML表格。"""
        html = ""
        if title:
            html += f"<h3>{title}</h3>"
        html += df.to_html(index=False, classes="data-table", border=0,
                          float_format="%.4f")
        return html

    def generate_html_report(self,
                             metadata: Dict,
                             regime_results: Optional[Dict] = None,
                             changepoint_results: Optional[Dict] = None,
                             volatility_results: Optional[Dict] = None,
                             charts: Optional[Dict[str, str]] = None,
                             output_path: str = "output/report.html") -> str:
        """
        生成完整的HTML报告。

        参数:
            metadata: 报告元数据（标的、日期范围等）
            regime_results: HMM状态检测结果
            changepoint_results: 变点检测结果
            volatility_results: 波动率分析结果
            charts: 图表字典（base64编码）
            output_path: 输出文件路径

        返回:
            HTML文件路径
        """
        if charts is None:
            charts = {}

        # 构建HTML内容
        html_parts = []

        # HTML头部
        html_parts.append(self._html_header(metadata))

        # 摘要
        html_parts.append(self._html_summary(metadata, regime_results,
                                             changepoint_results,
                                             volatility_results))

        # HMM状态分析
        if regime_results:
            html_parts.append(self._html_regime_section(regime_results, charts))

        # 变点检测
        if changepoint_results:
            html_parts.append(self._html_changepoint_section(changepoint_results,
                                                             charts))

        # 波动率分析
        if volatility_results:
            html_parts.append(self._html_volatility_section(volatility_results,
                                                            charts))

        # HTML尾部
        html_parts.append(self._html_footer())

        # 合并并写入文件
        full_html = "\n".join(html_parts)

        import os
        os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(full_html)

        return output_path

    def _html_header(self, metadata: Dict) -> str:
        """生成HTML头部和样式。"""
        return f"""
<!DOCTYPE html>
<html lang="zh-CN">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>市场状态检测报告 - {metadata.get('标的', 'N/A')}</title>
    <style>
        body {{
            font-family: "Microsoft YaHei", "PingFang SC", "Helvetica Neue", Arial, sans-serif;
            max-width: 1200px;
            margin: 0 auto;
            padding: 20px;
            background-color: #f5f6fa;
            color: #2c3e50;
            line-height: 1.6;
        }}
        .header {{
            background: linear-gradient(135deg, #2c3e50, #3498db);
            color: white;
            padding: 30px;
            border-radius: 10px;
            margin-bottom: 20px;
            text-align: center;
        }}
        .header h1 {{
            margin: 0;
            font-size: 28px;
        }}
        .header .subtitle {{
            margin-top: 10px;
            font-size: 14px;
            opacity: 0.9;
        }}
        .section {{
            background: white;
            padding: 25px;
            border-radius: 10px;
            margin-bottom: 20px;
            box-shadow: 0 2px 8px rgba(0,0,0,0.08);
        }}
        .section h2 {{
            color: #2c3e50;
            border-bottom: 3px solid #3498db;
            padding-bottom: 10px;
            margin-top: 0;
        }}
        .section h3 {{
            color: #34495e;
            margin-top: 20px;
        }}
        .chart-container {{
            text-align: center;
            margin: 20px 0;
        }}
        .chart-container img {{
            max-width: 100%;
            border-radius: 8px;
            box-shadow: 0 2px 6px rgba(0,0,0,0.1);
        }}
        .data-table {{
            width: 100%;
            border-collapse: collapse;
            margin: 15px 0;
            font-size: 13px;
        }}
        .data-table th {{
            background-color: #3498db;
            color: white;
            padding: 10px;
            text-align: left;
            font-weight: 600;
        }}
        .data-table td {{
            padding: 8px 10px;
            border-bottom: 1px solid #ecf0f1;
        }}
        .data-table tr:nth-child(even) {{
            background-color: #f8f9fa;
        }}
        .data-table tr:hover {{
            background-color: #e8f4fd;
        }}
        .metric-card {{
            display: inline-block;
            background: #ecf0f1;
            padding: 15px 25px;
            border-radius: 8px;
            margin: 5px;
            min-width: 150px;
            text-align: center;
        }}
        .metric-card .label {{
            font-size: 12px;
            color: #7f8c8d;
            margin-bottom: 5px;
        }}
        .metric-card .value {{
            font-size: 20px;
            font-weight: bold;
            color: #2c3e50;
        }}
        .regime-badge {{
            display: inline-block;
            padding: 5px 15px;
            border-radius: 20px;
            font-weight: bold;
            font-size: 14px;
        }}
        .footer {{
            text-align: center;
            padding: 20px;
            color: #95a5a6;
            font-size: 12px;
        }}
        .alert {{
            padding: 12px 20px;
            border-radius: 8px;
            margin: 10px 0;
        }}
        .alert-warning {{
            background-color: #fff3cd;
            border: 1px solid #ffeaa7;
            color: #856404;
        }}
        .alert-info {{
            background-color: #d1ecf1;
            border: 1px solid #b8daff;
            color: #0c5460;
        }}
    </style>
</head>
<body>
    <div class="header">
        <h1>市场状态智能检测报告</h1>
        <div class="subtitle">
            标的: {metadata.get('标的', 'N/A')} |
            数据范围: {metadata.get('开始日期', 'N/A')} ~ {metadata.get('结束日期', 'N/A')} |
            分析方法: {metadata.get('方法', 'N/A')} |
            生成时间: {metadata.get('生成时间', datetime.now().strftime('%Y-%m-%d %H:%M'))}
        </div>
    </div>
"""

    def _html_summary(self, metadata, regime, changepoint, volatility) -> str:
        """生成摘要部分。"""
        html = '<div class="section"><h2>分析摘要</h2>'

        # 当前状态
        if regime and "current_regime" in regime:
            cr = regime["current_regime"]
            html += '<div style="margin: 15px 0;">'
            html += f'<span class="regime-badge" style="background: {self.COLOR_SCHEME.get(cr["当前状态"], "#bdc3c7")}; color: white;">'
            html += f'当前市场状态: {cr["当前状态"]} (置信度: {cr["状态置信度"]})'
            html += '</span></div>'

        # 关键指标卡片
        html += '<div style="margin: 15px 0;">'
        if regime and "model_summary" in regime:
            html += f'<div class="metric-card"><div class="label">HMM状态数</div><div class="value">{regime["model_summary"]["状态数"]}</div></div>'
            html += f'<div class="metric-card"><div class="label">对数似然</div><div class="value">{regime.get("log_likelihood", 0):.1f}</div></div>'
        if changepoint and "summary" in changepoint:
            html += f'<div class="metric-card"><div class="label">检测变点数</div><div class="value">{changepoint["summary"]["总变点数"]}</div></div>'
        if volatility and "基本统计" in volatility:
            html += f'<div class="metric-card"><div class="label">年化波动率</div><div class="value">{volatility["基本统计"]["年化波动率"]*100:.1f}%</div></div>'
        html += '</div>'

        # 风险提示
        if regime and "current_regime" in regime:
            cr = regime["current_regime"]
            if cr["当前状态"] in ["熊市", "危机"]:
                html += '<div class="alert alert-warning">当前市场处于下行状态，建议关注风险控制。</div>'
            elif cr["当前状态"] == "震荡":
                html += '<div class="alert alert-info">当前市场处于震荡状态，建议等待趋势确认。</div>'

        html += '</div>'
        return html

    def _html_regime_section(self, regime, charts) -> str:
        """生成HMM状态分析部分。"""
        html = '<div class="section"><h2>HMM市场状态分析</h2>'

        # 状态统计表
        if "state_stats" in regime:
            html += self._table_to_html(regime["state_stats"], "状态统计特征")

        # 持续时间分析
        if "duration_analysis" in regime:
            html += self._table_to_html(regime["duration_analysis"], "状态持续时间分析")

        # 当前状态详情
        if "current_regime" in regime:
            cr = regime["current_regime"]
            html += '<h3>当前状态详情</h3>'
            html += '<table class="data-table"><tr><th>指标</th><th>值</th></tr>'
            html += f'<tr><td>当前状态</td><td>{cr["当前状态"]}</td></tr>'
            html += f'<tr><td>状态置信度</td><td>{cr["状态置信度"]}</td></tr>'
            html += '<tr><td>状态概率分布</td><td>'
            for state, prob in cr["状态概率分布"].items():
                html += f'{state}: {prob*100:.1f}%  '
            html += '</td></tr>'
            html += '<tr><td>下一期状态预测</td><td>'
            for state, prob in cr["下一期状态预测"].items():
                html += f'{state}: {prob*100:.1f}%  '
            html += '</td></tr>'
            html += '</table>'

        # 图表
        if "price_regime" in charts:
            html += f'<div class="chart-container"><img src="{charts["price_regime"]}"></div>'
        if "transition_heatmap" in charts:
            html += f'<div class="chart-container"><img src="{charts["transition_heatmap"]}"></div>'

        html += '</div>'
        return html

    def _html_changepoint_section(self, changepoint, charts) -> str:
        """生成变点检测部分。"""
        html = '<div class="section"><h2>变点检测分析</h2>'

        # 摘要
        if "summary" in changepoint:
            s = changepoint["summary"]
            html += '<div style="margin: 15px 0;">'
            html += f'<div class="metric-card"><div class="label">总变点数</div><div class="value">{s["总变点数"]}</div></div>'
            html += f'<div class="metric-card"><div class="label">CUSUM变点</div><div class="value">{s["CUSUM变点"]}</div></div>'
            html += f'<div class="metric-card"><div class="label">PELT变点</div><div class="value">{s["PELT变点"]}</div></div>'
            html += f'<div class="metric-card"><div class="label">贝叶斯变点</div><div class="value">{s["贝叶斯变点"]}</div></div>'
            html += '</div>'

        # 变点列表
        if "merged" in changepoint and changepoint["merged"]:
            cp_df = pd.DataFrame(changepoint["merged"])
            html += self._table_to_html(cp_df, "检测到的变点列表")

        # 图表
        if "changepoint_chart" in charts:
            html += f'<div class="chart-container"><img src="{charts["changepoint_chart"]}"></div>'

        html += '</div>'
        return html

    def _html_volatility_section(self, volatility, charts) -> str:
        """生成波动率分析部分。"""
        html = '<div class="section"><h2>波动率分析</h2>'

        # 基本统计
        if "基本统计" in volatility:
            bs = volatility["基本统计"]
            html += '<h3>波动率基本统计</h3>'
            html += '<table class="data-table"><tr><th>指标</th><th>值</th></tr>'
            html += f'<tr><td>日均收益率</td><td>{bs["日均收益率"]*100:.4f}%</td></tr>'
            html += f'<tr><td>日波动率</td><td>{bs["日波动率"]*100:.4f}%</td></tr>'
            html += f'<tr><td>年化波动率</td><td>{bs["年化波动率"]*100:.2f}%</td></tr>'
            html += f'<tr><td>偏度</td><td>{bs["偏度"]:.4f}</td></tr>'
            html += f'<tr><td>峰度</td><td>{bs["峰度"]:.4f}</td></tr>'
            html += f'<tr><td>最大单日涨幅</td><td>{bs["最大单日涨幅"]*100:.2f}%</td></tr>'
            html += f'<tr><td>最大单日跌幅</td><td>{bs["最大单日跌幅"]*100:.2f}%</td></tr>'
            html += '</table>'

        # GARCH参数
        if "GARCH参数" in volatility:
            gp = volatility["GARCH参数"]
            html += '<h3>GARCH(1,1)模型参数</h3>'
            html += '<table class="data-table"><tr><th>参数</th><th>值</th></tr>'
            for k, v in gp.items():
                if isinstance(v, float):
                    html += f'<tr><td>{k}</td><td>{v:.6f}</td></tr>'
                else:
                    html += f'<tr><td>{k}</td><td>{v}</td></tr>'
            html += '</table>'

        # 聚集效应
        if "聚集效应" in volatility:
            cl = volatility["聚集效应"]
            html += '<h3>波动率聚集效应</h3>'
            html += f'<p>存在波动率聚集: <strong>{"是" if cl["存在波动率聚集"] else "否"}</strong> '
            html += f'(Ljung-Box p值: {cl["Ljung-Box p值"]:.4f}, 聚集强度: {cl["聚集强度"]})</p>'
            if cl.get("GARCH持续性(alpha+beta)"):
                html += f'<p>GARCH持续性参数 (alpha+beta): <strong>{cl["GARCH持续性(alpha+beta)"]:.4f}</strong></p>'

        # 图表
        if "volatility_chart" in charts:
            html += f'<div class="chart-container"><img src="{charts["volatility_chart"]}"></div>'

        html += '</div>'
        return html

    def _html_footer(self) -> str:
        """生成HTML尾部。"""
        return """
    <div class="footer">
        <p>市场状态智能检测器 (Regime Detector) v1.0.0 | MIT License</p>
        <p>本报告由HMM和变点检测算法自动生成，仅供参考，不构成投资建议。</p>
    </div>
</body>
</html>
"""
