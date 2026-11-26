# FactorMiner Web

[简体中文](README.md) · [English documentation](../docs/README.en.md)

The React, TypeScript and Vite frontend provides the Dashboard, Launchpad, Inspector and Data Center for a single researcher running FactorMiner locally from a source checkout.

Use Node.js ≥22.12. Install and start the backend separately using the [product guide](../docs/FactorMiner_Product_Report_EN.md). The Python wheel supplies the CLI and API modules. The frontend and complete reference workspace come with the checkout.

From the root of the source tree, run:

```bash
cd web
npm ci
npm run dev -- --host 127.0.0.1 --port 5173 --strictPort
```

Open `http://localhost:5173`. The frontend connects to `http://localhost:8000` and `ws://localhost:8000/ws/monitor`, so the backend must listen on local port 8000.

From `web/`, run the frontend checks:

```bash
npm test
npm run build
```

The Node test runner exercises component and hook event handling. The build checks TypeScript and bundles assets in `dist/`. Run `npm run preview` to inspect the production build locally.

The [Web UI specification](WEB_UI_PRD.en.md) lists shipped and planned features. Task state lives in the backend process and is replayed on reconnect. Restarting the backend clears task history. Lifecycle labels organize research and never place trades.
