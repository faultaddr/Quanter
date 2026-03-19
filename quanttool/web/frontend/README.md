# QuantTool Frontend

基于 React 18 + Next.js 14 + TypeScript + TailwindCSS 构建的量化交易平台前端。

## 技术栈

- **框架**: Next.js 14 (App Router)
- **语言**: TypeScript 5.x
- **样式**: TailwindCSS 3.x
- **图表**: ECharts 5.x
- **状态管理**: Zustand 4.x
- **数据请求**: Axios + React Query

## 快速开始

### 安装依赖

```bash
cd quanttool/web/frontend
npm install
```

### 开发模式

```bash
npm run dev
```

访问 http://localhost:3000

### 生产构建

```bash
npm run build
npm start
```

## 项目结构

```
frontend/
├── app/                    # Next.js 页面路由
│   ├── layout.tsx         # 根布局
│   ├── page.tsx           # 首页(盘面概览)
│   ├── analyze/           # 股票分析
│   ├── backtest/          # 策略回测
│   ├── model/             # ML模型
│   ├── monitor/           # 实时监控
│   ├── scan/              # 智能选股
│   └── picks/             # 智能荐股
├── components/            # React 组件
│   ├── layout/           # 布局组件
│   ├── charts/           # 图表组件
│   ├── stock/            # 股票相关组件
│   ├── backtest/         # 回测相关组件
│   ├── model/            # 模型相关组件
│   └── ui/               # 通用 UI 组件
├── hooks/                 # 自定义 Hooks
├── stores/                # Zustand 状态管理
├── lib/                   # 工具库
│   ├── api/              # API 模块
│   ├── utils.ts          # 工具函数
│   └── constants.ts      # 常量定义
├── types/                 # TypeScript 类型定义
└── public/               # 静态资源
```

## 功能模块

### 1. 盘面概览
- 快速功能入口
- 市场指数实时展示
- 最近访问记录

### 2. 股票分析
- K线图表 (支持缩放、拖拽)
- 技术指标 (MACD/KDJ/RSI)
- 筹码分布图
- 交易信号面板

### 3. 策略回测
- 多策略选择
- 收益曲线对比
- 详细指标展示
- 交易记录查询

### 4. ML模型
- GBM模型训练
- 训练进度监控
- 模型管理
- 一键预测

### 5. 实时监控
- WebSocket 实时行情
- 自定义监控列表
- 自动刷新

### 6. 智能选股
- 多条件筛选
- 信号组合过滤
- 评分排序

### 7. 智能荐股
- 模型选择
- Top K 推荐
- 置信度展示

## API 代理配置

开发环境下，API 请求会代理到后端服务：

```javascript
// next.config.js
async rewrites() {
  return [
    {
      source: '/api/:path*',
      destination: 'http://localhost:8000/api/:path*',
    },
  ];
}
```

## 主题定制

主题颜色定义在 `tailwind.config.ts`：

```typescript
colors: {
  primary: '#3B82F6',    // 主色调
  success: '#10B981',    // 涨/买入
  danger: '#EF4444',     // 跌/卖出
  warning: '#F59E0B',    // 警告
  // ...
}
```

## 组件使用示例

```tsx
import { KlineChart, SignalPanel, MetricsGrid } from '@/components';

// K线图
<KlineChart data={klineData} height={400} showVolume showMA />

// 信号面板
<SignalPanel signals={signals} />

// 回测指标
<MetricsGrid metrics={backtestMetrics} />
```

## 环境要求

- Node.js 18+
- npm 9+ 或 yarn 1.22+

## License

MIT
