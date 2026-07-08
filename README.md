# 市场状态智能检测器 (Regime Detector)

基于隐马尔可夫模型(HMM)和变点检测算法的金融市场状态识别工具。

## 项目简介

本工具通过分析A股市场真实数据（akshare接口），自动识别市场的牛市、熊市、震荡、危机等状态切换。核心算法包括高斯隐马尔可夫模型（Baum-Welch训练 + Viterbi解码）、CUSUM/PELT/贝叶斯变点检测、以及GARCH(1,1)波动率建模，最终生成自包含的HTML可视化报告。

## 核心特性

- **真实数据驱动**：所有数据来自akshare接口，禁止任何随机或伪造数据
- **多算法融合**：HMM状态检测 + 三种变点检测算法 + GARCH波动率建模
- **灵活配置**：支持2-4状态HMM、8种技术特征、日/周/月频率
- **可视化报告**：HTML报告含价格状态着色图、转移概率热力图、变点标注图、波动率状态图
- **终端美化**：使用rich库显示分析进度和结果摘要

## 快速开始

### 安装

```bash
git clone <repository-url>
cd regime-detector
pip install -r requirements.txt
```

### 使用

```bash
# 检测个股市场状态
python detect.py --ticker 000001 --states 3 --start 20200101

# 检测指数（仅HMM）
python detect.py --index sh000300 --method hmm

# 全部分析方法
python detect.py --index sh000001 --method all --states 4 --start 20200101
```

运行后在 `output/` 目录生成HTML报告和状态CSV文件。

## 项目结构

```
regime-detector/
├── detect.py                    # CLI入口，协调整个分析流程
├── regime_detector/
│   ├── __init__.py              # 包初始化和模块导出
│   ├── data.py                  # akshare数据获取（指数/个股/多频率）
│   ├── features.py              # 特征工程（8个技术特征）
│   ├── hmm_model.py             # HMM隐马尔可夫模型（Baum-Welch/Viterbi）
│   ├── changepoint.py           # 变点检测（CUSUM/PELT/贝叶斯在线）
│   ├── volatility.py            # 波动率分析（GARCH/聚集效应/状态分类）
│   └── report.py                # HTML报告生成（matplotlib图表+base64嵌入）
├── SKILL.md                     # 技能文档（含FAQ和能力边界）
├── README.md                    # 项目说明
├── requirements.txt             # Python依赖
└── output/                      # 输出目录（HTML报告+CSV数据）
```

## 模块说明

### data.py - 数据获取

使用akshare获取真实A股数据：
- 指数数据：上证指数、沪深300、上证50、中证500、深证成指、创业板指、科创50
- 个股数据：前复权历史行情
- 多频率支持：日线、周线、月线（通过重采样实现）
- 自动计算对数收益率和滚动波动率

### features.py - 特征工程

构建HMM特征矩阵，包含8个技术特征：

| 特征 | 公式 | 经济含义 |
|------|------|----------|
| log_return | ln(P_t / P_{t-1}) | 对数收益率 |
| rolling_vol | std(r, window=20) | 滚动波动率 |
| squared_return | r_t^2 | GARCH条件方差代理 |
| volume_change | (V_t - V_{t-1}) / V_{t-1} | 成交量变化率 |
| turnover_indicator | Z-score(turnover) | 换手率相对活跃度 |
| momentum | P_t - P_{t-n} | 动量指标 |
| rsi | 100 - 100/(1+RS) | RSI相对强弱 |
| price_accel | r_t - r_{t-1} | 价格加速度 |

### hmm_model.py - HMM市场状态检测

基于hmmlearn的高斯HMM实现：
- **训练**：Baum-Welch算法（EM算法的HMM特例），最大化观测序列对数似然
- **解码**：Viterbi算法，动态规划求最优状态序列
- **后验概率**：前向-后向算法计算 P(z_t | X)
- **状态排序**：按收益率均值排序，赋予经济含义（牛/熊/震荡/危机）
- **状态统计**：均值、方差、出现频率、预期持续时间 E[T] = 1/(1-A[i,i])
- **可复现性**：random_state=42 保证初始化一致

### changepoint.py - 变点检测

三种独立算法：

1. **CUSUM**：累积和检测，通过累积偏离均值的偏差识别均值漂移
2. **PELT**（ruptures库）：精确线性时间算法，支持l1/l2/rbf/normal/ar代价函数
3. **贝叶斯在线变点**：基于Adams & MacKay (2007)，维护运行长度后验分布

### volatility.py - 波动率分析

- **GARCH(1,1)**：arch库拟合条件异方差模型，输出omega/alpha/beta参数
- **波动率状态分类**：基于分位数或GARCH条件波动率分为低/中/高波动
- **聚集效应**：Ljung-Box检验平方收益率自相关性，判断是否存在波动率聚集
- **实现vs预测**：对比历史实现波动率与GARCH条件波动率

### report.py - 报告生成

使用matplotlib生成4类图表，以base64嵌入HTML：
1. 价格走势+HMM状态着色图
2. 状态转移概率热力图
3. 变点检测标注图
4. 波动率状态图（含GARCH条件波动率）

## 参数详解

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| --ticker | str | - | 个股代码（如000001） |
| --index | str | - | 指数代码（如sh000300） |
| --method | str | all | hmm/changepoint/volatility/all |
| --states | int | 3 | HMM状态数（2/3/4） |
| --start | str | 20200101 | 开始日期 |
| --end | str | 今天 | 结束日期 |
| --freq | str | daily | daily/weekly/monthly |
| --output | str | output | 输出目录 |
| --features | str | 全部 | 逗号分隔的特征名 |

## 技术约束

1. 所有代码使用真实akshare数据，禁止np.random生成业务数据
2. HMM初始化使用random_state=42保证可复现（非业务数据生成）
3. 所有代码有详细中文注释
4. hmmlearn的GaussianHMM正确使用fit/predict/predict_proba
5. ruptures库正确使用PELT算法（Pelt类 + predict方法）

## 依赖版本

```
akshare>=1.12.0
hmmlearn>=0.3.0
ruptures>=1.1.0
arch>=5.0.0
numpy>=1.24.0
pandas>=2.0.0
scipy>=1.10.0
scikit-learn>=1.3.0
matplotlib>=3.7.0
rich>=13.0.0
```

## 许可证

MIT License

## 免责声明

本工具仅供学术研究和教育目的使用，不构成任何投资建议。金融市场分析存在固有风险，使用者需自行承担投资决策的全部责任。
