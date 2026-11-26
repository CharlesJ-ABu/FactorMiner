# Local services, exchange connectivity and optional proxies

[简体中文](vpn_setup.md) · [English documentation](../README.en.md)

Start with the source installation in the [product guide](../FactorMiner_Product_Report_EN.md). FactorMiner binds to loopback by default and has no complete authentication or multi-tenant system. A working local API, access to an exchange, availability of a market and completeness of local data are separate checks.

## Check the local API and external market data

After starting the backend, query its health endpoint:

```bash
curl http://localhost:8000/api/health
```

The expected response, `{"status":"ok","engine":"FactorMinerDirector"}`, confirms that the local API can respond. The check does not contact an exchange or LLM provider.

Data Center uses `/api/exchange_meta` on port 8000. This request attempts to retrieve exchange metadata:

```bash
curl 'http://localhost:8000/api/exchange_meta?exchange=binance&trade_type=futures'
```

If the request fails upstream, the API may return a static list. An HTTP 200 response or a populated dropdown alone cannot confirm exchange connectivity. For a download, inspect its final state (`completed`, `failed` or `partial_failed`) and error details. Progress can reach 100% even when a download failed.

## Optional HTTP proxy

Leave proxy variables unset if a proxy is unnecessary. For public market data, the default downloader reads `HTTP_PROXY` and `HTTPS_PROXY` from the process environment. If you have an HTTP proxy, substitute its actual address and port for the example below:

```bash
export HTTP_PROXY='http://127.0.0.1:7890'
export HTTPS_PROXY='http://127.0.0.1:7890'
export NO_PROXY='localhost,127.0.0.1'
```

Start the CLI or backend in that terminal. Restart a running backend after changing the variables. This behavior applies to the public-market path through `DataDownloader.get_exchange_instance()` without a `config_id`. Account-specific exchange clients and LLM requests may have separate settings.

Creating a `.env` file alone does not load its variables. Keep proxy credentials and API keys out of public configurations, documentation and commits. This guide does not endorse a VPN provider. Changing networks does not guarantee access to a given exchange or symbol.

## Failures and offline use

- Record the actual error and check its exchange, market, CCXT symbol, timeframe and UTC interval.
- Distinguish a timeout from an access restriction, an unsupported market or missing bars. A static market list cannot supply missing prices.
- With complete local Feather data, regular GP, RL and NumPy NN runs can work offline. Missing files may trigger an automatic download. Live LLM generation still needs a model service.
- Remote deployments require their own API and WebSocket addresses, proxy setup and access controls. The installation procedure covers the local workbench.

Exchange account settings are first created on save in `config/user_config/config.json` under the current working directory. This also applies to wheel installations. The writer uses a temporary file and atomic replacement. On POSIX, new files use mode `0600`. Restrict access to backups that may contain credentials and keep them outside the public repository.
