# Enhanced DataFetcher 集成说明

## 概述

我们已成功将您提供的Tushare令牌和EastMoney Cookie集成为EnhancedDataFetcher类，该类继承自QuantTool的标准IDataProvider接口。

## 主要特性

1. **双数据源支持**：同时支持Tushare和EastMoney数据源
2. **智能回退机制**：当EastMoney不可用时自动回退到Tushare
3. **标准接口兼容**：完全兼容QuantTool的数据提供程序架构
4. **安全凭证管理**：支持环境变量和硬编码凭证两种方式

## 数据源

### Tushare
- Token: `744295f7af6adf63074518f919f5ad5054caf8b84d3c07c066f5c42e`
- 支持完整的A股历史数据查询
- 适用于每日K线、基本面数据等

### EastMoney
- Cookie: 包含您提供的完整Cookie字符串
- 提供实时性较强的数据
- 作为Tushare的补充数据源

## 使用方法

### 1. 直接使用预设凭证
```python
from quanttool.infrastructure.data_providers.data_fetcher import create_data_fetcher_with_credentials

# 创建已配置好所有凭证的实例
fetcher = create_data_fetcher_with_credentials()
fetcher.initialize()

# 获取数据
data = fetcher.get_bars(['000001.SZ'], start_date, end_date)
```

### 2. 手动配置
```python
from quanttool.infrastructure.data_providers.data_fetcher import EnhancedDataFetcher

fetcher = EnhancedDataFetcher(
    tushare_token="your_tushare_token",
    eastmoney_cookie="your_eastmoney_cookie"
)
```

## 优势

1. **统一API**：单一接口访问两个数据源
2. **高可用性**：多源回退保证数据可用性
3. **兼容性**：无缝集成到现有的QuantTool架构
4. **灵活性**：支持自定义数据源优先级

## 注册名称

此数据提供程序在系统中注册为 `enhanced_data_fetcher`，可以通过注册表获取：

```python
from quanttool.core.registry import registry, ComponentType
fetcher = registry.create(ComponentType.DATA_PROVIDER, "enhanced_data_fetcher")
```

## 示例

参见 `/examples/enhanced_data_fetcher_example.py` 文件，其中包含了完整的使用示例。