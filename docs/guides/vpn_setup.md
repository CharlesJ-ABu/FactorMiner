# 网络、代理与本机服务说明

先完成[产品指南](../FactorMiner_Product_Report_ZH.md)中的源码安装。FactorMiner 默认只监听本机，当前没有完整认证和多租户体系。网络可达性、交易所是否提供指定市场和本地行情是否完整是不同条件。

## 区分本机 API 与外部行情

启动本机后端后可以检查：

```bash
curl http://localhost:8000/api/health
```

正常响应为 `{"status":"ok","engine":"FactorMinerDirector"}`，只表示本机 API 能响应，不证明交易所或 LLM 可达。

Data Center 的元数据接口是 `/api/exchange_meta`，使用 8000 端口。例如手动查询会尝试访问交易所：

```bash
curl 'http://localhost:8000/api/exchange_meta?exchange=binance&trade_type=futures'
```

元数据失败时可能返回静态名单，因此 HTTP 200 或下拉框有选项不是网络成功证据。下载结果应以 `completed / failed / partial_failed` 和具体错误判断，100% 进度不代表成功。

## 可选 HTTP 代理

无需代理时不要设置代理变量。若已有可用 HTTP 代理，默认公开行情下载路径会读取启动进程的 `HTTP_PROXY` 和 `HTTPS_PROXY`。以下端口仅为示例，应替换为实际配置：

```bash
export HTTP_PROXY='http://127.0.0.1:7890'
export HTTPS_PROXY='http://127.0.0.1:7890'
export NO_PROXY='localhost,127.0.0.1'
```

在同一终端启动 CLI 或后端，变量变更后重启后端才能生效。该说明对应 `DataDownloader.get_exchange_instance()` 不带 config_id 的公开行情路径；不能据此推断所有客户端、配置账户路径或 LLM 请求都共用这套代理。

项目不会因为创建 `.env` 就自动加载变量。不要把代理凭证或 API 密钥写入公开配置、文档或提交。本文不提供 VPN 服务商推荐，也不保证更换网络会使特定交易所/标的可用。

## 失败与离线使用

- 检查错误中的交易所、市场、CCXT symbol、周期和 UTC 区间，保留具体异常。
- 网络超时、访问限制、无此市场和缺失 K 线是不同原因；静态名单不会补出行情。
- 已有完整本地 Feather 行情时，普通 GP/RL/NumPy NN 计算可离线运行。缺文件可能触发自动下载；真实 LLM 生成仍需模型服务。
- 远程部署需另外处理前端固定 API/WS 地址、网络代理和访问控制，不属于当前本机安装验收。

交易所账户配置保存在运行工作目录下的 `config/user_config/config.json`，首次实际保存时创建；wheel 安装也不会写入 site-packages。保存使用临时文件和原子替换，在 POSIX 上新文件权限为 0600。该文件可能包含凭证，备份时单独限制访问，不纳入公开源码。
