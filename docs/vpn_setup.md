# VPN设置指南

## 概述

由于网络限制，某些加密货币交易所的API在某些地区可能无法直接访问。本指南将帮助您配置VPN以正常访问交易所API。

## 为什么需要VPN？

### 网络限制
- **Binance**: 在某些地区被限制访问
- **OKX**: 在某些地区被限制访问
- **Bybit**: 在某些地区被限制访问

### 影响
- 无法获取实时交易对列表
- 无法下载最新市场数据
- 数据管理功能受限

## VPN解决方案

### 1. 商业VPN服务

#### 推荐的VPN服务商
- **ExpressVPN**: 速度快，支持多地区服务器
- **NordVPN**: 价格实惠，安全性高
- **Surfshark**: 无限设备连接
- **CyberGhost**: 用户友好，适合新手

#### 设置步骤
1. 注册并下载VPN客户端
2. 安装并登录VPN客户端
3. 选择支持加密货币交易所的地区服务器（如：香港、新加坡、美国）
4. 连接VPN
5. 测试网络连接

### 2. 自建VPN服务器

#### 使用VPS搭建
```bash
# 使用Docker快速搭建
docker run -d \
  --name vpn \
  --restart=always \
  -p 1194:1194/udp \
  -e VPN_USER=user \
  -e VPN_PASS=password \
  kylemanna/openvpn
```

#### 使用云服务商
- **AWS Lightsail**: 简单易用
- **DigitalOcean**: 价格实惠
- **Vultr**: 全球节点多

### 3. 代理服务器

#### HTTP代理
```python
import requests

proxies = {
    'http': 'http://proxy-server:port',
    'https': 'http://proxy-server:port'
}

response = requests.get('https://api.binance.com/api/v3/exchangeInfo', proxies=proxies)
```

#### SOCKS5代理
```python
import requests

proxies = {
    'http': 'socks5://proxy-server:port',
    'https': 'socks5://proxy-server:port'
}

response = requests.get('https://api.binance.com/api/v3/exchangeInfo', proxies=proxies)
```

## 测试连接

### 1. 测试网络连通性

```bash
# 测试Binance API连接
curl -I https://api.binance.com/api/v3/exchangeInfo

# 测试OKX API连接
curl -I https://www.okx.com/api/v5/public/instruments

# 测试Bybit API连接
curl -I https://api.bybit.com/v5/market/instruments-info
```

### 2. 测试CCXT连接

```python
import ccxt

# 测试Binance连接
try:
    exchange = ccxt.binance()
    markets = exchange.load_markets()
    print(f"✅ Binance连接成功，获取到 {len(markets)} 个交易对")
except Exception as e:
    print(f"❌ Binance连接失败: {e}")

# 测试OKX连接
try:
    exchange = ccxt.okx()
    markets = exchange.load_markets()
    print(f"✅ OKX连接成功，获取到 {len(markets)} 个交易对")
except Exception as e:
    print(f"❌ OKX连接失败: {e}")

# 测试Bybit连接
try:
    exchange = ccxt.bybit()
    markets = exchange.load_markets()
    print(f"✅ Bybit连接成功，获取到 {len(markets)} 个交易对")
except Exception as e:
    print(f"❌ Bybit连接失败: {e}")
```

### 3. 测试FactorMiner API

```bash
# 测试交易对API
curl -s http://localhost:8080/api/data/symbols/binance | jq '.note'

# 如果返回null，说明连接成功
# 如果返回"使用预设交易对（网络连接失败）"，说明需要VPN
```

## 配置FactorMiner使用代理

### 1. 环境变量配置

```bash
# 设置HTTP代理
export HTTP_PROXY=http://proxy-server:port
export HTTPS_PROXY=http://proxy-server:port

# 设置SOCKS5代理
export HTTP_PROXY=socks5://proxy-server:port
export HTTPS_PROXY=socks5://proxy-server:port
```

### 2. Python代码配置

```python
import ccxt
import requests

# 配置代理
proxies = {
    'http': 'http://proxy-server:port',
    'https': 'http://proxy-server:port'
}

# 创建交易所实例时使用代理
exchange = ccxt.binance({
    'enableRateLimit': True,
    'timeout': 10000,
    'proxies': proxies
})
```

### 3. 修改FactorMiner配置

在 `webui/routes/data_api.py` 中添加代理支持：

```python
@bp.route('/symbols/<exchange>', methods=['GET'])
def get_symbols(exchange):
    """获取指定交易所的交易对列表"""
    try:
        import ccxt
        
        # 配置代理（如果需要）
        proxies = {
            'http': os.environ.get('HTTP_PROXY'),
            'https': os.environ.get('HTTPS_PROXY')
        }
        
        # 创建交易所实例
        exchange_class = getattr(ccxt, exchange)
        exchange_instance = exchange_class({
            'enableRateLimit': True,
            'timeout': 10000,
            'proxies': proxies if any(proxies.values()) else None
        })
        
        # 其余代码...
```

## 常见问题

### 1. VPN连接后仍然无法访问

**可能原因：**
- VPN服务器被交易所封禁
- DNS解析问题
- 防火墙阻止

**解决方案：**
- 尝试不同的VPN服务器
- 更换DNS服务器（如8.8.8.8, 1.1.1.1）
- 检查防火墙设置

### 2. 连接速度慢

**优化建议：**
- 选择地理位置较近的VPN服务器
- 使用有线网络连接
- 关闭不必要的网络应用

### 3. 安全性考虑

**注意事项：**
- 使用可信的VPN服务商
- 避免在VPN连接时进行敏感操作
- 定期更换VPN密码

## 替代方案

### 1. 使用其他数据源

如果无法使用VPN，可以考虑：
- 使用本地已下载的数据
- 使用其他数据提供商
- 手动导入数据文件

### 2. 离线模式

FactorMiner支持离线模式，使用预设的交易对列表：
- 包含常用的加密货币交易对
- 支持现货和期货交易
- 可以正常进行因子挖掘分析

## 总结

配置VPN是访问加密货币交易所API的最佳解决方案。通过本指南，您应该能够：

1. 选择合适的VPN服务
2. 正确配置网络连接
3. 测试API连通性
4. 在FactorMiner中使用真实数据

如果遇到问题，请参考常见问题部分或联系技术支持。 