# 代理池使用指南

## 概述

代理池模块提供了完整的代理管理功能，帮助规避反爬虫检测。

## 功能特点

- ✅ 多种代理来源：文件、列表、API
- ✅ 自动健康检查
- ✅ 智能轮换策略（基于成功率）
- ✅ 失败自动切换
- ✅ 支持付费代理API（快代理、芝麻代理等）

## 快速开始

### 1. 从文件加载代理

```python
from quanttool.infrastructure.data_providers.anti_crawler import setup_proxy_pool

# 创建代理文件 proxies.txt
# 格式：每行一个代理
# http://1.2.3.4:8080
# socks5://user:pass@5.6.7.8:1080
# 10.0.0.1:3128

pool = setup_proxy_pool(proxy_file='proxies.txt')
```

### 2. 从列表加载代理

```python
from quanttool.infrastructure.data_providers.anti_crawler import setup_proxy_pool

proxy_list = [
    'http://1.2.3.4:8080',
    'http://user:pass@5.6.7.8:3128',
    'socks5://10.0.0.1:1080'
]

pool = setup_proxy_pool(proxy_list=proxy_list)
```

### 3. 使用付费代理API

```python
from quanttool.infrastructure.data_providers.anti_crawler import setup_proxy_pool

# 快代理示例
pool = setup_proxy_pool(
    api_url='https://dps.kdlapi.com/api/getdps/',
    api_key='your_api_key'  # 或设置环境变量 PROXY_API_KEY
)

# 芝麻代理示例
pool = setup_proxy_pool(
    api_url='http://webapi.http.zhimacangku.com/getip',
    api_key='your_api_key'
)
```

### 4. 在 EnhancedDataFetcher 中使用

```python
from quanttool.infrastructure.data_providers.data_fetcher import EnhancedDataFetcher

fetcher = EnhancedDataFetcher(
    tushare_token='your_token',
    use_proxy=True,
    proxy_file='proxies.txt',  # 或
    proxy_list=['http://1.2.3.4:8080'],  # 或
    proxy_api_url='https://api.proxy.com/get',
    proxy_api_key='your_key'
)
```

## 代理文件格式

```
# 注释以 # 开头
# 格式1: protocol://user:pass@host:port
http://user:password@192.168.1.1:8080
socks5://proxy.example.com:1080

# 格式2: host:port (默认http)
10.0.0.1:3128
172.16.0.1:8888
```

## 推荐代理服务商

| 服务商 | 特点 | 价格参考 |
|--------|------|----------|
| 快代理 | 国内节点多，稳定 | ¥100/月起 |
| 芝麻代理 | 按量计费，灵活 | ¥0.04/个起 |
| 阿布云 | 企业级，高可用 | ¥500/月起 |
| 站大爷 | 便宜，适合小规模 | ¥30/月起 |

## 免费代理（不推荐）

免费代理不稳定，仅用于测试：

```python
# 免费代理示例（需自行抓取更新）
free_proxies = [
    'http://123.45.67.89:8080',
    # ...
]
pool = setup_proxy_pool(proxy_list=free_proxies)

# 健康检查
stats = pool.health_check()
print(f"存活代理: {stats['alive']}/{stats['checked']}")
```

## 最佳实践

1. **使用付费代理**：免费代理不稳定，容易被封
2. **设置合理延迟**：即使使用代理，也要有延迟（0.5-2秒）
3. **定期健康检查**：移除失效代理
4. **备用数据源**：代理失败时有备用方案

```python
# 推荐配置
fetcher = EnhancedDataFetcher(
    tushare_token='your_token',
    use_proxy=True,
    proxy_api_url='your_proxy_api',
    max_workers=3,  # 降低并发
)

# 代理池会自动管理：
# - 成功率高的代理优先使用
# - 失败自动切换
# - 冷却机制（失败多次后暂时停用）
```
