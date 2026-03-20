# QuantTool - A 股量化交易分析平台

[中文](./README.md) | [English](./README_EN.md)

QuantTool 是一個專業的 A 股量化交易分析平台，提供技術分析、因子研究、策略回測、風險控制等核心功能。

## 核心特性

- **免費資料來源**：優先使用 Ashare、EastMoney、AkShare 等免費資料來源
- **即時資料**：支援分鐘級即時行情取得
- **Web 介面**：現代化的 Web 前端，支援全功能操作
- **策略回測**：支援多種技術指標策略回測，真實模擬 A 股交易規則
- **因子研究**：IC/IR 分析、因子優化、因子中性化
- **風控管理**：行業揭露監控、黑名單檢查、倉位收縮

## 核心功能

### 1. 多維度技術分析評分系統

基於三大類因子的分層評分架構：

```
最終評分 = 趨勢得分 × 位置修正係數
```

#### 趨勢因子（權重 40%）
| 因子 | 權重 | 說明 |
|------|------|------|
| 趨勢強度 | 20% | MA20 乖離率，DMI 狀態修正 |
| 均線斜率 | 20% | MA5 斜率判斷趨勢方向 |
| MACD 動量 | 20% | MACD 柱狀圖變化 |
| 資金流向 | 20% | OBV 資金流評分 |
| 成交量 | 10% | 量價配合度 |
| K 線形態 | 10% | 位置敏感評分 |

### 2. K 線形態識別系統

支援識別 20+ 種經典 K 線形態，包括錘子線、吞沒形態、晨星、暮星等。

### 3. 因子研究與優化

| 功能 | 說明 |
|------|------|
| 因子有效性檢驗 | IC/IR 分析，評估因子預測能力 |
| 因子權重優化 | IR 加權、IC 加權、等權、風險平價 |
| 因子中性化 | 行業中性、市值中性 |
| 因子流水線處理 | Winsorize、Standardize 處理 |

### 4. 組合風控管理

| 功能 | 說明 |
|------|------|
| 行業揭露監控 | 單一行業倉位限制（預設 20%） |
| 黑名單檢查 | 禁止持倉股票監控 |
| 倉位收縮 | 基於回撤動態調整倉位 |
| 風險評分 | 多維度風險評估 |

### 5. A 股交易約束

| 約束類型 | 說明 |
|----------|------|
| 漲跌停限制 | 漲停不能買入，跌停不能賣出 |
| ST 股限制 | 可設定排除 ST 股票 |
| 真實交易成本 | 佣金、印花稅、滑點模擬 |

### 6. 內建交易策略

| 策略名稱 | 類型 | 說明 |
|----------|------|------|
| `ma_cross` | 趨勢追蹤 | 均線交叉策略 |
| `dual_ma` | 趨勢追蹤 | 雙均線策略 |
| `breakout` | 突破策略 | 價格突破 N 日高低點 |
| `turtle` | 趨勢追蹤 | 海龜交易策略 |
| `ma_alignment` | 趨勢追蹤 | 均線多頭排列策略 |
| `rsi` | 震盪指標 | RSI 超買超賣策略 |
| `macd` | 趨勢指標 | MACD 金叉死叉策略 |
| `kdj` | 震盪指標 | KDJ 金叉死叉策略 |
| `bollinger` | 震盪指標 | 布林帶回歸策略 |

## 技術架構

```
QuantTool/
├── quanttool/
│   ├── core/                    # 核心功能
│   │   ├── errors.py           # 錯誤處理
│   │   ├── logging.py          # 日誌
│   │   └── registry.py         # 組件註冊
│   │
│   ├── domain/                  # 領域層
│   │   ├── interfaces/         # 介面定義
│   │   └── models/             # 資料模型
│   │
│   ├── application/             # 應用服務層
│   │   ├── analysis_service.py
│   │   ├── backtest_service.py
│   │   └── factor_service.py
│   │
│   ├── infrastructure/          # 基礎設施層
│   │   ├── data_providers/     # 資料提供者
│   │   │   ├── ashare_provider.py
│   │   │   ├── akshare_minute_provider.py
│   │   │   └── data_fetcher.py
│   │   └── stores/             # 儲存層
│   │
│   ├── strategies/              # 交易策略
│   │   ├── ma_cross.py
│   │   ├── breakout.py
│   │   └── ...
│   │
│   ├── factors/                # 因子庫
│   │   ├── factor_validator.py
│   │   ├── factor_pipeline.py
│   │   ├── factor_registry.py
│   │   └── neutralizer.py
│   │
│   ├── optimization/           # 優化器
│   │   └── weight_optimizer.py
│   │
│   ├── risk/                   # 風險管理
│   │   └── risk_controller.py
│   │
│   ├── backtest/               # 回測引擎
│   │   ├── engine.py
│   │   └── ashare_constraints.py
│   │
│   ├── web/                    # Web 層
│   │   ├── api/               # API 路由
│   │   └── frontend/          # Next.js 前端
│   │
│   └── cli/                    # 命令列工具
│       └── main.py
│
└── tests/                      # 測試用例
```

## 安裝

### 前置要求

- Python 3.9+
- Node.js 18+
- npm 或 yarn

### 安裝步驟

```bash
# 複製專案
git clone https://github.com/faultaddr/Quanter.git
cd Quanter

# 安裝 Python 依賴
pip install -e .

# 安裝前端依賴
cd quanttool/web/frontend
npm install
```

## 快速開始

### 啟動服務

```bash
# 啟動後端服務
uvicorn quanttool.web.app:app --host 0.0.0.0 --port 8000

# 啟動前端開發伺服器
cd quanttool/web/frontend
npm run dev
```

訪問 http://localhost:3000 開啟 Web 介面。

### 使用 Web 介面

| 頁面 | 功能 |
|------|------|
| `/` | 盤面概覽、市場指數 |
| `/analyze` | 股票分析、K 線、技術指標 |
| `/backtest` | 策略回測、收益對比 |
| `/factors` | 因子研究、IC/IR 分析 |
| `/risk` | 組合風控、風險檢查 |
| `/scan` | 智慧選股、條件篩選 |
| `/picks` | AI 推薦股票 |
| `/monitor` | 即時行情監控 |
| `/model` | ML 模型訓練預測 |

### 使用 CLI

```bash
# 分析股票
quant analyze 600519 --days 360

# 回測策略
quant backtest run --strategy ma_cross --symbol 600519 \
  --start 2023-01-01 --end 2024-01-01 --cash 100000
```

### 使用 Python API

```python
from quanttool.factors.stock_analyzer import StockAnalyzer
from quanttool.backtest.engine import BacktestEngine

# 分析股票
analyzer = StockAnalyzer()
report = analyzer.analyze_stock("600519", days=360)
print(report.summary)

# 回測策略
engine = BacktestEngine()
result = engine.run(
    symbol="600519",
    strategy="ma_cross",
    start_date="2023-01-01",
    end_date="2024-01-01",
    initial_capital=1000000,
)
print(f"收益率: {result.total_return:.2%}")
```

## 資料來源優先順序

1. **Ashare** - 免費、無需 Token，主力資料來源
2. **EastMoney** - 免費、資料豐富
3. **AkShare** - 免費、介面豐富
4. **TuShare** - 需要 Token，作為備選

## 效能指標

| 操作 | P50 | P95 |
|------|-----|-----|
| 快取命中 | < 10ms | < 50ms |
| 資料取得 | < 500ms | < 2s |
| 完整分析 | < 2s | < 5s |

## 測試

```bash
# 執行所有測試
pytest tests/ -v

# 執行覆蓋率測試
pytest tests/ --cov=quanttool --cov-report=html
```

目前測試覆蓋：400+ 測試用例通過。

## 授權條款

MIT License
