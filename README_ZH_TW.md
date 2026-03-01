# QuantTool - A股市量化交易分析平台

QuantTool 是一個專為A股市場設計的量化分析平台，提供技術分析、形態識別、多因子評分、風險熔斷等核心功能。

## 核心功能

### 1. 多維度技術分析評分系統

基於三大類因子的分層評分架構：

```
最終評分 = 趨勢得分 × 位置修正係數
```

#### 趨勢因子（權重40%）
| 因子 | 權重 | 說明 |
|------|------|------|
| 趨勢強度 | 20% | MA20乖離率，DMI狀態修正 |
| 均線斜率 | 20% | MA5斜率判斷趨勢方向 |
| MACD動量 | 20% | MACD柱狀圖變化 |
| 資金流向 | 20% | OBV資金流評分 |
| 成交量 | 10% | 量價配合度 |
| K線形態 | 10% | 位置敏感評分 |

### 2. K線形態識別系統

支援識別20+種經典K線形態：

#### 單根形態
- **看漲形態**：錘子線、倒錘子、大陽線
- **看跌形態**：吊頸線、流星線、大陰線
- **中性形態**：十字星、長上影線、長下影線、紡錘線

#### 組合形態
- **看漲組合**：看漲吞沒、晨星、穿刺線
- **看跌組合**：看跌吞沒、暮星、烏雲蓋頂

#### 形態視覺化
- ASCII藝術K線圖（彩色顯示：🟢綠色陽線/🔴紅色陰線）
- 形態示意圖解說明

### 3. 智能熔斷機制

多層風險控制體系：

| 熔斷規則 | 觸發條件 | 效果 |
|----------|----------|------|
| 強力看跌熔斷 | 大陰線+高位 | 評分降至≤25分，倉位0% |
| 高位看跌熔斷 | 高位+看跌形態 | 評分降至≤25分，倉位0% |
| 誘多陷阱熔斷 | 高位+看漲+極端超買 | 強制觀望 |
| 極端超買熔斷 | 3+指標同時爆表 | 位置係數≤0.30 |

### 4. 內建交易策略

| 策略名稱 | 類型 | 說明 |
|----------|------|------|
| `ma_cross` | 趨勢追蹤 | 均線交叉策略（金叉買入，死叉賣出） |
| `dual_ma` | 趨勢追蹤 | 雙均線策略（支援自定義週期） |
| `breakout` | 突破策略 | 價格突破N日高低點 |
| `turtle` | 趨勢追蹤 | 海龜交易策略（唐奇安通道） |
| `ma_alignment` | 趨勢追蹤 | 均線多頭排列策略 |
| `rsi` | 震盪指標 | RSI超買超賣策略 |
| `macd` | 趨勢指標 | MACD金叉死叉策略 |
| `kdj` | 震盪指標 | KDJ金叉死叉策略 |
| `bollinger` | 震盪指標 | 布林帶回歸策略 |

### 5. 四部分報告架構

```
第一部分：核心結論區
├── 操作指令（買入/賣出/觀望）
├── 技術評分（0-100分）
├── 置信度評估
└── 關鍵理由

第二部分：多維信號共振分析
├── 趨勢狀態（DMI+均線排列）
├── 動能狀態（MACD+RSI）
├── 位置狀態（布林帶+CCI+WR）
├── 形態特權區（K線形態定性）
└── K線視覺化圖表

第三部分：量化評分與因子拆解
├── 綜合評分計算公式
├── 各因子得分明細
├── 位置修正係數
└── 紅黑榜

第四部分：交易執行計畫
├── 策略類型判定
├── 入場/止損/目標位
├── 倉位建議
└── 風險提示
```

## 安裝

```bash
git clone https://github.com/yourusername/quanttool.git
cd quanttool
pip install -e .
```

## 快速開始

### 1. 配置數據源

```bash
export TUSHARE_TOKEN="your_tushare_token"
```

### 2. 分析股票

```bash
# 分析單只股票
quant analyze 000001.SZ

# 指定週期
quant analyze 000001.SZ --days 360

# 市場掃描
quant analyze scan --market all --days 360 --top 10
```

### 3. 回測策略

```bash
# 均線交叉策略
quant backtest run --strategy ma_cross --symbol 000001.SZ \
  --start 2023-01-01 --end 2023-06-01 --cash 100000

# RSI策略
quant backtest run --strategy rsi --symbol 000001.SZ \
  --start 2023-01-01 --end 2023-06-01

# 海龜策略
quant backtest run --strategy turtle --symbol 000001.SZ \
  --start 2023-01-01 --end 2023-06-01
```

## 專案架構

```
quanttool/
├── factors/                    # 因子分析模組
│   ├── scoring_system.py       # 多維度評分系統
│   ├── candlestick_patterns.py # K線形態識別
│   ├── stock_analyzer.py       # 股票綜合分析
│   └── tech_indicators.py      # 技術指標計算
├── strategies/                 # 交易策略模組
│   ├── ma_cross.py            # 均線交叉策略
│   ├── dual_ma.py             # 雙均線策略
│   ├── breakout.py            # 突破策略
│   ├── turtle.py              # 海龜策略
│   ├── ma_alignment.py        # 均線排列策略
│   ├── rsi.py                 # RSI策略
│   ├── macd.py                # MACD策略
│   ├── kdj.py                 # KDJ策略
│   └── bollinger.py           # 布林帶策略
├── infrastructure/             # 基礎設施層
│   ├── data_providers/        # 數據提供者
│   └── stores/                # 儲存層
├── reports/                    # 報告生成
├── cli/                        # 命令列介面
└── web/                        # Web API
```

## Python API

```python
from quanttool.factors.stock_analyzer import StockAnalyzer
from quanttool.factors.candlestick_patterns import (
    CandlestickPatternRecognizer,
    draw_candlestick_chart,
    draw_pattern_illustration
)

# 分析股票
analyzer = StockAnalyzer()
report = analyzer.analyze_stock("000001.SZ", days=360)
print(report)

# 識別K線形態
recognizer = CandlestickPatternRecognizer()
patterns = recognizer.recognize_all_patterns(df, lookback=5)

# 繪製K線圖
chart = draw_candlestick_chart(df, num_candles=10)
print(chart)

# 獲取形態示意圖
illustration = draw_pattern_illustration("錘子線")
print(illustration)
```

## 報告範例

```markdown
## 第一部分：核心結論

### 🟢 操作指令：買入

**技術評分：70.5分（良好）**

### 📊 近期K線圖

最高: ¥63.77
   │        │    │  ████ │     │
   │        │    │  ████ │     │
   │   │    ████ │  ████ ▓▓▓▓  │
   │   │    ████ │  ████ ▓▓▓▓  │
最低: ¥60.27

> 圖例：🟢 綠色 = 陽線（漲） | 🔴 紅色 = 陰線（跌）
```

## 授權條款

MIT License