# FactorMiner Web

React、TypeScript 与 Vite 前端，提供 Dashboard、Launchpad、Inspector 和 Data Center。当前用于源码 checkout 中的本机单用户研究流程。

需要 Node.js ≥22.12。后端需另行安装、启动，步骤见[产品指南](../docs/FactorMiner_Product_Report_ZH.md)。Python wheel 提供 CLI/API 模块，不包含前端站点或完整工作区。

从源码根目录进入前端：

```bash
cd web
npm ci
npm run dev -- --host 127.0.0.1 --port 5173 --strictPort
```

访问 `http://localhost:5173`。API 和 WebSocket 当前分别使用 `http://localhost:8000`、`ws://localhost:8000/ws/monitor`，后端应监听本机 8000。

在 `web/` 中运行检查：

```bash
npm test
npm run build
```

测试使用 Node 内建 test runner 验证组件/hook 事件逻辑；构建检查 TypeScript 并由 Vite 打包至 `dist/`。`npm run preview` 可本地检查构建产物，不等于远程部署。

页面能力与规划见 [Web PRD](WEB_UI_PRD.md)。任务状态只在后端进程内保存；连接恢复会重放，服务重启不恢复。生命周期标签仅用于研究管理，不触发交易。
