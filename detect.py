#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
市场状态智能检测器 - CLI入口 (detect.py)
=========================================
命令行工具，整合数据获取、特征工程、HMM状态检测、
变点检测和波动率分析，生成HTML可视化报告。

使用示例:
    # 检测个股市场状态（3状态HMM）
    python detect.py --ticker 000001 --states 3 --start 20200101

    # 检测沪深300指数（仅变点检测）
    python detect.py --index sh000300 --method changepoint

    # 全部分析方法
    python detect.py --index sh000001 --method all --states 4 --start 20200101
"""

import argparse
import os
import sys
import json
import numpy as np
import pandas as pd
from datetime import datetime

# 添加项目根目录到路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from regime_detector import (
    DataFetcher, FeatureEngineer, HMMRegimeDetector,
    ChangepointDetector, VolatilityAnalyzer, ReportGenerator,
)
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TimeRemainingColumn
from rich.table import Table
from rich.panel import Panel
from rich import box

console = Console()


def parse_args():
    """解析命令行参数。"""
    parser = argparse.ArgumentParser(
        description="市场状态智能检测器 - 基于HMM和变点检测的金融市场状态识别工具",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
使用示例:
  python detect.py --ticker 000001 --states 3 --start 20200101
  python detect.py --index sh000300 --method hmm
  python detect.py --index sh000001 --method all --states 4 --start 20200101
  python detect.py --ticker 600519 --method volatility --start 20210101
        """,
    )

    # 标的选择（互斥组）
    target_group = parser.add_mutually_exclusive_group(required=True)
    target_group.add_argument(
        "--ticker", type=str, default=None,
        help="个股代码，如 000001（平安银行）、600519（贵州茅台）",
    )
    target_group.add_argument(
        "--index", type=str, default=None,
        help="指数代码，如 sh000001（上证指数）、sh000300（沪深300）",
    )

    # 分析参数
    parser.add_argument(
        "--method", type=str, default="all",
        choices=["hmm", "changepoint", "volatility", "all"],
        help="分析方法: hmm(状态检测), changepoint(变点检测), volatility(波动率), all(全部)",
    )
    parser.add_argument(
        "--states", type=int, default=3,
        choices=[2, 3, 4],
        help="HMM状态数量: 2(牛/熊), 3(牛/震荡/熊), 4(牛/震荡/熊/危机)",
    )
    parser.add_argument(
        "--start", type=str, default="20200101",
        help="开始日期，格式 YYYYMMDD，默认 20200101",
    )
    parser.add_argument(
        "--end", type=str, default=None,
        help="结束日期，格式 YYYYMMDD，默认今天",
    )
    parser.add_argument(
        "--freq", type=str, default="daily",
        choices=["daily", "weekly", "monthly"],
        help="数据频率: daily(日线), weekly(周线), monthly(月线)",
    )
    parser.add_argument(
        "--output", type=str, default="output",
        help="输出目录，默认 output/",
    )
    parser.add_argument(
        "--features", type=str, default=None,
        help="指定HMM特征（逗号分隔），默认使用全部特征。"
             "可选: log_return,rolling_vol,squared_return,volume_change,"
             "turnover_indicator,momentum,rsi,price_accel",
    )

    return parser.parse_args()


def run_hmm_analysis(df: pd.DataFrame, feature_eng: FeatureEngineer,
                     n_states: int, feature_list=None) -> dict:
    """
    运行HMM市场状态检测分析。

    参数:
        df: 含特征的DataFrame
        feature_eng: 特征工程器
        n_states: 状态数
        feature_list: 特征列表

    返回:
        HMM分析结果字典
    """
    # 构建特征矩阵
    feature_matrix, used_features, dates = feature_eng.build_feature_matrix(
        df, feature_list=feature_list, scale=True
    )

    # 创建并训练HMM
    hmm = HMMRegimeDetector(n_states=n_states, n_iter=100, random_state=42)
    hmm.fit(feature_matrix, feature_names=used_features)

    # 预测状态
    states = hmm.predict(feature_matrix)

    # 获取后验概率
    proba = hmm.predict_proba(feature_matrix)

    # 获取转移矩阵
    transmat = hmm.get_transition_matrix()

    # 状态统计
    state_stats = hmm.get_state_statistics(feature_matrix, used_features)

    # 持续时间分析
    duration_analysis = hmm.get_state_duration_analysis()

    # 当前状态
    current_regime = hmm.get_current_regime(feature_matrix)

    # 模型评分
    log_likelihood = hmm.score(feature_matrix)

    # 将状态序列对齐到原始DataFrame
    full_states = np.full(len(df), -1)
    valid_mask = df[["date"] + used_features].notna().all(axis=1).values
    full_states[valid_mask] = states

    return {
        "states": full_states,
        "states_clean": states,
        "dates": dates,
        "probabilities": proba,
        "transition_matrix": transmat,
        "state_labels": hmm.state_labels,
        "state_stats": state_stats,
        "duration_analysis": duration_analysis,
        "current_regime": current_regime,
        "log_likelihood": log_likelihood,
        "model_summary": hmm.get_model_summary(),
        "feature_names": used_features,
    }


def run_changepoint_analysis(returns: np.ndarray) -> dict:
    """
    运行变点检测分析。

    参数:
        returns: 收益率序列

    返回:
        变点检测结果字典
    """
    detector = ChangepointDetector(threshold=3.0, min_size=20)
    results = detector.detect_all(returns, methods=["cusum", "pelt", "bayesian"])
    return results


def run_volatility_analysis(returns: np.ndarray) -> dict:
    """
    运行波动率分析。

    参数:
        returns: 收益率序列

    返回:
        波动率分析结果字典
    """
    analyzer = VolatilityAnalyzer(window=20)

    # 拟合GARCH
    analyzer.fit_garch(returns, p=1, q=1)

    # 获取综合摘要
    summary = analyzer.get_volatility_summary(returns)

    # 获取波动率状态
    vol_states, thresholds = analyzer.classify_volatility_states(
        returns, method="garch"
    )

    # 获取实现波动率
    realized_vol = analyzer.compute_realized_volatility(returns)

    # 获取GARCH条件波动率
    garch_vol = analyzer.conditional_vol if analyzer.conditional_vol is not None else None

    summary["vol_states"] = vol_states
    summary["realized_vol"] = realized_vol
    summary["garch_vol"] = garch_vol
    summary["thresholds"] = thresholds

    return summary


def generate_charts(df: pd.DataFrame, regime_results: dict,
                    changepoint_results: dict, volatility_results: dict,
                    method: str) -> dict:
    """
    生成所有可视化图表。

    参数:
        df: 原始数据DataFrame
        regime_results: HMM结果
        changepoint_results: 变点结果
        volatility_results: 波动率结果
        method: 分析方法

    返回:
        图表字典（base64编码）
    """
    report_gen = ReportGenerator()
    charts = {}

    returns = df["log_return"].dropna().values

    # HMM相关图表
    if method in ["hmm", "all"] and regime_results:
        # 价格+状态着色图
        valid_mask = regime_results["states"] >= 0
        df_plot = df[valid_mask].reset_index(drop=True)
        states_plot = regime_results["states"][valid_mask]

        if len(df_plot) > 0:
            charts["price_regime"] = report_gen.plot_price_with_regimes(
                df_plot, states_plot, regime_results["state_labels"]
            )

        # 转移概率热力图
        charts["transition_heatmap"] = report_gen.plot_transition_heatmap(
            regime_results["transition_matrix"],
            regime_results["state_labels"]
        )

    # 变点检测图表
    if method in ["changepoint", "all"] and changepoint_results:
        charts["changepoint_chart"] = report_gen.plot_changepoints(
            df, returns, changepoint_results.get("merged", [])
        )

    # 波动率图表
    if method in ["volatility", "all"] and volatility_results:
        charts["volatility_chart"] = report_gen.plot_volatility_states(
            df, returns,
            volatility_results.get("vol_states", np.ones(len(returns), dtype=int)),
            volatility_results.get("realized_vol", np.zeros(len(returns))),
            volatility_results.get("garch_vol")
        )

    return charts


def display_results(regime_results, changepoint_results, volatility_results,
                    method, console):
    """在终端显示分析结果摘要。"""
    # HMM结果
    if method in ["hmm", "all"] and regime_results:
        console.print(Panel("[bold cyan]HMM市场状态检测结果[/bold cyan]",
                            box=box.ROUNDED))

        # 当前状态
        cr = regime_results["current_regime"]
        console.print(f"  当前状态: [bold]{cr['当前状态']}[/bold] "
                      f"(置信度: {cr['状态置信度']})")

        # 状态概率分布
        console.print("  状态概率分布:")
        for state, prob in cr["状态概率分布"].items():
            bar = "█" * int(prob * 20)
            console.print(f"    {state}: {prob*100:5.1f}% {bar}")

        # 状态统计表
        table = Table(title="状态统计摘要", box=box.ROUNDED)
        table.add_column("状态", style="cyan")
        table.add_column("样本数", justify="right")
        table.add_column("频率", justify="right")
        table.add_column("预期持续", justify="right")

        stats = regime_results["state_stats"]
        dur = regime_results["duration_analysis"]
        for i in range(len(stats)):
            table.add_row(
                stats.iloc[i]["状态"],
                str(stats.iloc[i]["样本数"]),
                stats.iloc[i]["出现频率"],
                dur.iloc[i]["预期持续时间(期)"],
            )
        console.print(table)

        # 转移矩阵
        console.print("\n  状态转移概率矩阵:")
        transmat = regime_results["transition_matrix"]
        labels = regime_results["state_labels"]
        header = "          " + "  ".join(f"{l:>6s}" for l in labels)
        console.print(f"  [dim]{header}[/dim]")
        for i, label in enumerate(labels):
            row = "  ".join(f"{transmat[i,j]:6.3f}" for j in range(len(labels)))
            console.print(f"  {label:>6s}  {row}")

    # 变点结果
    if method in ["changepoint", "all"] and changepoint_results:
        console.print(Panel("[bold magenta]变点检测结果[/bold magenta]",
                            box=box.ROUNDED))
        s = changepoint_results["summary"]
        console.print(f"  总变点数: [bold]{s['总变点数']}[/bold]")
        console.print(f"  CUSUM变点: {s['CUSUM变点']}")
        console.print(f"  PELT变点: {s['PELT变点']}")
        console.print(f"  贝叶斯变点: {s['贝叶斯变点']}")

        if changepoint_results.get("merged"):
            table = Table(title="检测到的变点", box=box.ROUNDED)
            table.add_column("位置", style="magenta")
            table.add_column("类型")
            table.add_column("算法")
            table.add_column("置信度", justify="right")
            for cp in changepoint_results["merged"][:10]:
                table.add_row(
                    str(cp["位置"]),
                    cp.get("类型", "N/A"),
                    cp.get("算法", "N/A"),
                    f"{cp.get('置信度', 0):.3f}",
                )
            console.print(table)

    # 波动率结果
    if method in ["volatility", "all"] and volatility_results:
        console.print(Panel("[bold yellow]波动率分析结果[/bold yellow]",
                            box=box.ROUNDED))
        bs = volatility_results["基本统计"]
        console.print(f"  年化波动率: [bold]{bs['年化波动率']*100:.2f}%[/bold]")
        console.print(f"  日均收益率: {bs['日均收益率']*100:.4f}%")
        console.print(f"  偏度: {bs['偏度']:.4f}")
        console.print(f"  峰度: {bs['峰度']:.4f}")

        if "GARCH参数" in volatility_results:
            gp = volatility_results["GARCH参数"]
            console.print(f"\n  GARCH(1,1)参数:")
            console.print(f"    omega: {gp['omega（长期方差）']:.6f}")
            console.print(f"    alpha: {gp['alpha[1]（ARCH系数）']:.6f}")
            console.print(f"    beta: {gp['beta[1]（GARCH系数）']:.6f}")
            console.print(f"    持续性(alpha+beta): {gp['alpha+beta（持续性）']:.4f}")
            console.print(f"    平稳: {'是' if gp['是否平稳'] else '否'}")

        if "聚集效应" in volatility_results:
            cl = volatility_results["聚集效应"]
            console.print(f"\n  波动率聚集: {'存在' if cl['存在波动率聚集'] else '不存在'} "
                          f"(p值: {cl['Ljung-Box p值']:.4f}, 强度: {cl['聚集强度']})")


def main():
    """主函数，协调整个分析流程。"""
    args = parse_args()

    # 确定标的
    if args.index:
        target = args.index
        target_type = "指数"
    else:
        target = args.ticker
        target_type = "个股"

    # 打印启动信息
    console.print(Panel.fit(
        f"[bold]市场状态智能检测器[/bold]\n"
        f"标的: {target_type} {target}\n"
        f"方法: {args.method}\n"
        f"状态数: {args.states}\n"
        f"数据范围: {args.start} ~ {args.end or '今天'}\n"
        f"频率: {args.freq}",
        border_style="cyan",
    ))

    # 创建输出目录
    output_dir = args.output
    os.makedirs(output_dir, exist_ok=True)

    # 解析特征列表
    feature_list = None
    if args.features:
        feature_list = [f.strip() for f in args.features.split(",")]

    # 使用rich进度条
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TimeRemainingColumn(),
        console=console,
    ) as progress:

        # Step 1: 获取数据
        task1 = progress.add_task("[cyan]获取市场数据...", total=100)
        fetcher = DataFetcher()
        try:
            df = fetcher.fetch_data(
                ticker=args.ticker,
                index=args.index,
                start_date=args.start,
                end_date=args.end,
                freq=args.freq,
            )
        except Exception as e:
            console.print(f"[red]数据获取失败: {e}[/red]")
            sys.exit(1)
        progress.update(task1, completed=100)
        console.print(f"  获取数据: {len(df)} 条记录")

        # Step 2: 特征工程
        task2 = progress.add_task("[cyan]计算技术特征...", total=100)
        feature_eng = FeatureEngineer()
        df = feature_eng.compute_features(df)
        progress.update(task2, completed=100)
        console.print(f"  特征计算完成: {len(feature_eng.feature_names or [])} 个特征")

        # 准备收益率数据
        returns = df["log_return"].dropna().values

        # Step 3: 执行分析
        regime_results = None
        changepoint_results = None
        volatility_results = None

        if args.method in ["hmm", "all"]:
            task3 = progress.add_task("[cyan]训练HMM模型 (Baum-Welch)...", total=100)
            try:
                regime_results = run_hmm_analysis(
                    df, feature_eng, args.states, feature_list
                )
                progress.update(task3, completed=100)
                console.print(f"  HMM训练完成: 对数似然={regime_results['log_likelihood']:.1f}")
            except Exception as e:
                progress.update(task3, completed=100)
                console.print(f"  [red]HMM分析失败: {e}[/red]")

        if args.method in ["changepoint", "all"]:
            task4 = progress.add_task("[magenta]执行变点检测...", total=100)
            try:
                changepoint_results = run_changepoint_analysis(returns)
                progress.update(task4, completed=100)
                console.print(f"  变点检测完成: "
                              f"{changepoint_results['summary']['总变点数']} 个变点")
            except Exception as e:
                progress.update(task4, completed=100)
                console.print(f"  [red]变点检测失败: {e}[/red]")

        if args.method in ["volatility", "all"]:
            task5 = progress.add_task("[yellow]拟合GARCH模型...", total=100)
            try:
                volatility_results = run_volatility_analysis(returns)
                progress.update(task5, completed=100)
                console.print(f"  波动率分析完成: "
                              f"年化波动率={volatility_results['基本统计']['年化波动率']*100:.2f}%")
            except Exception as e:
                progress.update(task5, completed=100)
                console.print(f"  [red]波动率分析失败: {e}[/red]")

        # Step 4: 生成图表
        task6 = progress.add_task("[green]生成可视化图表...", total=100)
        charts = generate_charts(
            df, regime_results or {}, changepoint_results or {},
            volatility_results or {}, args.method
        )
        progress.update(task6, completed=100)
        console.print(f"  图表生成完成: {len(charts)} 张")

        # Step 5: 生成HTML报告
        task7 = progress.add_task("[green]生成HTML报告...", total=100)
        report_gen = ReportGenerator()

        # 准备元数据
        metadata = {
            "标的": f"{target_type} {target} ({df['name'].iloc[0] if 'name' in df.columns else target})",
            "开始日期": args.start,
            "结束日期": args.end or datetime.now().strftime("%Y%m%d"),
            "方法": args.method,
            "生成时间": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        }

        # 生成报告
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_path = os.path.join(output_dir, f"report_{target}_{timestamp}.html")

        report_gen.generate_html_report(
            metadata=metadata,
            regime_results=regime_results,
            changepoint_results=changepoint_results,
            volatility_results=volatility_results,
            charts=charts,
            output_path=report_path,
        )
        progress.update(task7, completed=100)

        # 保存状态数据为CSV
        if regime_results:
            csv_path = os.path.join(output_dir, f"states_{target}_{timestamp}.csv")
            export_df = df[["date", "close", "log_return"]].copy()
            export_df["hmm_state"] = regime_results["states"]
            valid_mask = export_df["hmm_state"] >= 0
            if valid_mask.any():
                state_labels = regime_results["state_labels"]
                export_df["state_label"] = "N/A"
                for i, label in enumerate(state_labels):
                    export_df.loc[export_df["hmm_state"] == i, "state_label"] = label
            export_df.to_csv(csv_path, index=False, encoding="utf-8-sig")
            console.print(f"  状态数据已保存: {csv_path}")

    # 显示终端结果
    console.print()
    display_results(regime_results, changepoint_results, volatility_results,
                    args.method, console)

    # 完成提示
    console.print()
    console.print(Panel.fit(
        f"[bold green]分析完成![/bold green]\n"
        f"HTML报告: {report_path}\n"
        f"输出目录: {os.path.abspath(output_dir)}",
        border_style="green",
    ))


if __name__ == "__main__":
    main()
