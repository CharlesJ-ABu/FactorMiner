import assert from 'node:assert/strict';
import fs from 'node:fs';
import { createRequire } from 'node:module';
import test from 'node:test';
import * as downloadStatus from '../src/utils/downloadStatus.ts';

const { downloadOutcome, emptyDownloadState, reduceDownloadProgress } = downloadStatus;

const event = (status, progress = 100, task_id = 'job-1') => ({
  type: 'download_progress', task_id, status, progress, message: `Result: ${status}`,
});
const running = { status: 'running', progress: 0, message: 'Starting...' };

for (const [status, tone] of [['completed', 'success'], ['failed', 'error'], ['partial_failed', 'warning']]) {
  test(`${status} selects the matching terminal display`, () => {
    const state = reduceDownloadProgress(running, event(status), 'job-1');
    assert.equal(state.status, status);
    assert.equal(state.progress, 100);
    assert.equal(downloadOutcome(state.status).tone, tone);
    if (tone !== 'success') assert.doesNotMatch(downloadOutcome(state.status).message, /Successfully/);
  });
}

test('100 percent without explicit successful terminal status remains running', () => {
  for (const status of ['running', undefined, 'unrecognized']) {
    const state = reduceDownloadProgress(running, event(status), 'job-1');
    assert.equal(state.status, 'running');
    assert.equal(downloadOutcome(state.status), null);
  }
});

test('an unrelated task or no accepted task cannot change the displayed result', () => {
  assert.equal(reduceDownloadProgress(running, event('completed', 100, 'job-2'), 'job-1'), running);
  assert.equal(reduceDownloadProgress(emptyDownloadState, event('completed'), null), emptyDownloadState);
});

test('late progress cannot turn a terminal failure back into running or success', () => {
  const failed = reduceDownloadProgress(running, event('partial_failed'), 'job-1');
  assert.equal(reduceDownloadProgress(failed, event('running', 80), 'job-1'), failed);
  assert.equal(reduceDownloadProgress(failed, event('completed'), 'job-1'), failed);
});

// Run the actual component and WebSocket hook with controlled hooks and delivery.
// No DOM, server or network is needed; events can arrive without an intervening render.
const { transformSync } = createRequire(import.meta.url)('esbuild');
function loadSource(relativePath, resolve, globals = {}) {
  const source = fs.readFileSync(new URL(relativePath, import.meta.url), 'utf8');
  const { code } = transformSync(source, { loader: relativePath.endsWith('tsx') ? 'tsx' : 'ts', format: 'cjs', jsx: 'automatic' });
  const module = { exports: {} };
  new Function('require', 'module', 'exports', ...Object.keys(globals), code)(resolve, module, module.exports, ...Object.values(globals));
  return module.exports;
}

async function downloadPage({ component = 'DataDownload', initialTasks = [], deferInitialTasks = false } = {}) {
  const states = [], effects = [], callbacks = [], refs = [], requests = [], sockets = [];
  const taskRequests = [];
  const timers = new Map();
  let si = 0, ei = 0, ci = 0, ri = 0, dirty = true, tree, queued = [], socket;
  let nextTimer = 0, mounted = true, updatesAfterUnmount = 0, socketUrl, latestHook;
  const same = (a, b) => a && b && a.length === b.length && a.every((v, i) => Object.is(v, b[i]));
  const hooks = {
    useState(initial) {
      const i = si++;
      if (!(i in states)) states[i] = typeof initial === 'function' ? initial() : initial;
      return [states[i], next => {
        if (!mounted) { updatesAfterUnmount++; return; }
        const value = typeof next === 'function' ? next(states[i]) : next;
        if (!Object.is(value, states[i])) { states[i] = value; dirty = true; }
      }];
    },
    useRef(initial) { const i = ri++; return refs[i] ?? (refs[i] = { current: initial }); },
    useEffect(fn, deps) {
      const i = ei++;
      if (!same(effects[i]?.deps, deps)) queued.push(() => {
        effects[i]?.cleanup?.(); effects[i] = { deps, cleanup: fn() };
      });
    },
    useCallback(fn, deps) {
      const i = ci++;
      if (!same(callbacks[i]?.deps, deps)) callbacks[i] = { deps, fn };
      return callbacks[i].fn;
    },
  };
  const fakeFetch = url => {
    if (url.endsWith('/api/download_data')) return new Promise((resolve, reject) => requests.push({ resolve, reject }));
    if (url.endsWith('/api/tasks')) {
      if (deferInitialTasks) return new Promise(resolve => taskRequests.push(resolve));
      return Promise.resolve({ json: async () => ({ tasks: initialTasks }) });
    }
    if (url.endsWith('/api/miners')) return Promise.resolve({ json: async () => ({ miners: ['GP'] }) });
    if (url.endsWith('/api/configs')) return Promise.resolve({ json: async () => ({ configs: { 'test.json': {} } }) });
    const data = url.includes('/api/exchange_meta')
      ? { symbols: ['BTC/USDT', 'ETH/USDT'], timeframes: ['1m', '1d'], trade_types: ['spot', 'futures'], min_date: '2017-01-01' }
      : { results: [] };
    return Promise.resolve({ ok: true, json: async () => data });
  };
  const globals = {
    setTimeout(fn, delay) { const id = ++nextTimer; timers.set(id, { fn, delay }); return id; },
    clearTimeout(id) { timers.delete(id); },
    console: { ...console, log: () => {}, error: () => {} },
  };
  const rawWebsocketHook = loadSource('../src/hooks/useWebSocket.ts', () => hooks, {
    ...globals, WebSocket: class {
      static OPEN = 1;
      constructor(url) { this.url = url; this.readyState = 0; this.closeCalls = 0; this.sent = []; socket = this; sockets.push(this); }
      open() { this.readyState = 1; this.onopen?.(); }
      close() { this.closeCalls++; if (this.readyState === 3) return; this.readyState = 3; this.onclose?.(); }
      send(message) { this.sent.push(message); }
    },
  });
  const websocketHook = {
    useWebSocket(url, autoReconnect, onMessage) {
      latestHook = rawWebsocketHook.useWebSocket(socketUrl ?? url, autoReconnect, onMessage);
      return latestHook;
    },
  };
  const Page = loadSource(`../src/pages/${component}.tsx`, name => {
    if (name === 'react') return hooks;
    if (name === 'react-router-dom') return { Link: 'Link' };
    if (name === 'echarts-for-react') return () => null;
    if (name === 'react/jsx-runtime') return { jsx: (type, props) => ({ type, props }), jsxs: (type, props) => ({ type, props }) };
    if (name === 'lucide-react') return { Download: 'Download', CheckCircle2: 'CheckCircle2', AlertCircle: 'AlertCircle', Info: 'Info', Loader2: 'Loader2' };
    if (name === '../hooks/useWebSocket') return websocketHook;
    if (name === '../utils/downloadStatus') return downloadStatus;
    if (name === 'rc-slider') return () => null;
    if (name.endsWith('.css')) return {};
    throw new Error(`Unexpected import: ${name}`);
  }, { ...globals, fetch: fakeFetch })[component];
  function render() {
    if (!mounted) return;
    for (let pass = 0; pass < 25 && dirty; pass++) {
      dirty = false; si = ei = ci = ri = 0; queued = []; tree = Page();
      for (const effect of queued) effect();
    }
    assert.equal(dirty, false, 'render must stabilize');
  }
  async function flush() { for (let i = 0; i < 8; i++) { await Promise.resolve(); render(); } }
  function text(node) {
    if (typeof node === 'string' || typeof node === 'number') return String(node);
    if (Array.isArray(node)) return node.map(text).join(' ');
    return node && typeof node === 'object' ? text(node.props?.children) : '';
  }
  function flatten(node) {
    if (!node || typeof node !== 'object') return [];
    if (Array.isArray(node)) return node.flatMap(flatten);
    return [node, ...flatten(node.props?.children)];
  }
  render(); await flush();
  return {
    click(label) {
      const button = flatten(tree).find(node => node.type === 'button' && text(node).includes(label));
      assert.ok(button, `button missing: ${label}`);
      button.props.onClick(); render();
    },
    emit(message) { socket.onmessage({ data: JSON.stringify(message) }); },
    async accept(taskId = 'job-1') {
      requests.at(-1).resolve({ ok: true, json: async () => ({ task_id: taskId }) }); await flush();
    },
    async fail(kind) {
      if (kind === 'network') requests.at(-1).reject(new Error('Network fixture failure'));
      else requests.at(-1).resolve({ ok: kind !== 'http', status: 503, json: async () => ({}) });
      await flush();
    },
    async resolveTasks(tasks) {
      assert.ok(taskRequests.length, 'an initial tasks request must be pending');
      taskRequests.shift()({ json: async () => ({ tasks }) });
      await flush();
    },
    selectTask(taskId) {
      render();
      const row = flatten(tree).find(node => node.type === 'tr' && text(node).includes(taskId));
      assert.ok(row?.props.onClick, `task row missing: ${taskId}`);
      row.props.onClick(); render();
    },
    taskRow(taskId) {
      render();
      return text(flatten(tree).find(node => node.type === 'tr' && text(node).includes(taskId)));
    },
    scatter() {
      render();
      return flatten(tree).find(node => node.props?.option?.series)?.props.option.series[0].data;
    },
    flush,
    screen() { render(); return text(tree); },
    sockets,
    connection() { render(); return latestHook; },
    changeSocketUrl(url) { socketUrl = url; dirty = true; render(); },
    pendingReconnects() { return [...timers.values()].filter(timer => timer.delay === 3000).length; },
    runReconnectTimers() {
      for (const [id, timer] of [...timers]) {
        if (timer.delay === 3000) { timers.delete(id); timer.fn(); }
      }
      render();
    },
    unmount() { mounted = false; effects.forEach(effect => effect.cleanup?.()); },
    updatesAfterUnmount() { return updatesAfterUnmount; },
  };
}

test('page receives every event before the submit response, including a terminal followed by another task', async () => {
  const page = await downloadPage();
  page.click('Execute Batch Download');
  // No render between these WebSocket events: lastMessage alone would lose the terminal.
  page.emit(event('failed'));
  page.emit({ type: 'task_update', task: { id: 'other-task', progress: 50 } });
  page.emit(event('completed', 100, 'job-2'));
  page.emit(event('running', 80));
  await page.accept();
  assert.match(page.screen(), /Batch download failed\. Review/);
  assert.match(page.screen(), /Start another batch/);
  assert.doesNotMatch(page.screen(), /Successfully completed batch/);
  page.emit(event('completed'));
  await page.flush();
  assert.match(page.screen(), /Batch download failed\. Review/);
});

test('normal page submission isolates the accepted task and waits for explicit success', async () => {
  const page = await downloadPage();
  page.click('Execute Batch Download');
  await page.accept();
  page.emit(event('failed', 100, 'job-2'));
  page.emit(event('running', 100));
  await page.flush();
  assert.doesNotMatch(page.screen(), /Successfully completed batch|Batch download failed\. Review/);
  page.emit(event('completed'));
  await page.flush();
  assert.match(page.screen(), /Successfully completed batch download!/);
});

test('page replays the latest pending progress then displays a partial failure', async () => {
  const page = await downloadPage();
  page.click('Execute Batch Download');
  page.emit(event('running', 10));
  page.emit(event('running', 25));
  await page.accept();
  assert.match(page.screen(), /25\s*%/);
  page.emit(event('partial_failed'));
  await page.flush();
  assert.match(page.screen(), /Batch download partially failed/);
});

for (const kind of ['network', 'http', 'missing_id']) {
  test(`submission ${kind} failure clears buffered events and permits a clean retry`, async () => {
    const page = await downloadPage();
    page.click('Execute Batch Download');
    page.emit(event('completed'));
    await page.fail(kind);
    assert.match(page.screen(), /Batch download failed\. Review/);
    assert.match(page.screen(), /Network fixture failure|Download request failed \(503\)|Download task ID was not returned/);
    page.click('Start another batch');
    page.click('Execute Batch Download');
    await page.accept('retry-job');
    page.emit(event('failed'));
    page.emit(event('completed', 100, 'retry-job'));
    await page.flush();
    assert.match(page.screen(), /Successfully completed batch download!/);
  });
}

test('unmount closes its socket and ignores captured late events without reconnecting', async () => {
  const page = await downloadPage();
  const socket = page.sockets[0];
  socket.open();
  await page.flush();
  const late = { open: socket.onopen, message: socket.onmessage, close: socket.onclose, error: socket.onerror };
  page.unmount();
  assert.equal(socket.closeCalls, 1);
  assert.equal(socket.readyState, 3);
  late.open();
  late.message({ data: JSON.stringify(event('completed')) });
  late.close();
  late.error(new Error('late socket error'));
  page.runReconnectTimers();
  assert.equal(page.updatesAfterUnmount(), 0);
  assert.equal(page.pendingReconnects(), 0);
  assert.equal(page.sockets.length, 1);
});

test('URL changes close the old socket and only the current connection may deliver events', async () => {
  const page = await downloadPage();
  page.click('Execute Batch Download');
  await page.accept();
  const previous = page.sockets[0];
  previous.open();
  await page.flush();
  const lateMessage = previous.onmessage;
  const lateClose = previous.onclose;
  page.changeSocketUrl('ws://offline-fixture/new-monitor');
  assert.equal(previous.closeCalls, 1);
  const current = page.sockets[1];
  assert.equal(current.url, 'ws://offline-fixture/new-monitor');
  current.open();
  lateMessage({ data: JSON.stringify(event('failed')) });
  lateClose();
  await page.flush();
  assert.equal(page.connection().isConnected, true);
  assert.equal(page.pendingReconnects(), 0);
  assert.doesNotMatch(page.screen(), /Batch download failed\. Review/);
  page.emit(event('completed'));
  await page.flush();
  assert.match(page.screen(), /Successfully completed batch download!/);
  page.connection().sendMessage({ probe: 'current connection' });
  assert.deepEqual(current.sent, ['{"probe":"current connection"}']);
  assert.deepEqual(previous.sent, []);
  page.unmount();
});

test('a live page reconnects after a close, but unmount cancels a pending retry', async () => {
  const page = await downloadPage();
  const first = page.sockets[0];
  first.open();
  await page.flush();
  first.close();
  await page.flush();
  assert.equal(page.connection().isConnected, false);
  assert.equal(page.pendingReconnects(), 1);
  page.runReconnectTimers();
  assert.equal(page.sockets.length, 2);
  const second = page.sockets[1];
  second.open();
  await page.flush();
  assert.equal(page.connection().isConnected, true);
  second.close();
  assert.equal(page.pendingReconnects(), 1);
  page.unmount();
  assert.equal(page.pendingReconnects(), 0);
  page.runReconnectTimers();
  assert.equal(page.sockets.length, 2);
  assert.equal(page.updatesAfterUnmount(), 0);
});

const miningTerminals = [
  ['completed', /✅ 完成/], ['failed', /失败/], ['completed_empty', /已完成（无有效因子）/],
];

for (const [status, displayed] of miningTerminals) {
  test(`Launchpad preserves ${status} before another task event without an intervening render`, async () => {
    const initial = { id: 'mining-A', status: 'running', progress: 25, miner: 'GP', config: 'test.json', logs: [] };
    const page = await downloadPage({ component: 'Launchpad', initialTasks: [initial] });
    assert.match(page.screen(), /运行中 \(\s*25\s*%\)/);
    page.emit({ type: 'task_update', task: { ...initial, status, progress: 100 } });
    page.emit({ type: 'log', task_id: 'mining-B', text: 'another task log' });
    await page.flush();
    assert.match(page.screen(), displayed);
    assert.doesNotMatch(page.screen(), /运行中 \(\s*25\s*%\)/);
    page.unmount();
  });
}

for (const [status, displayed] of miningTerminals) {
  test(`Launchpad keeps ${status} and live tasks when the initial HTTP snapshot arrives late`, async () => {
    const initial = { id: 'mining-A', status: 'running', progress: 25, miner: 'GP', config: 'test.json', logs: [] };
    const page = await downloadPage({ component: 'Launchpad', deferInitialTasks: true });
    page.emit({ type: 'task_update', task: initial });
    await page.flush();
    page.selectTask(initial.id);
    page.emit({ type: 'task_update', task: { ...initial, status, progress: 100 } });
    page.emit({ type: 'task_update', task: { ...initial, id: 'live-B', progress: 50 } });
    page.emit({ type: 'log', task_id: initial.id, text: 'owned task log' });
    page.emit({ type: 'scatter', task_id: initial.id, epoch: 1, ic: 0.2, complexity: 3 });
    await page.flush();
    assert.match(page.taskRow(initial.id), displayed);
    await page.resolveTasks([initial, { ...initial, id: 'history-C', status: 'completed', progress: 100 }]);
    assert.match(page.taskRow(initial.id), displayed);
    assert.match(page.taskRow('live-B'), /运行中 \(\s*50\s*%\)/);
    assert.match(page.taskRow('history-C'), /✅ 完成/);
    assert.match(page.screen(), new RegExp(`Status:\\s*${status}`));
    assert.match(page.screen(), /owned task log/);
    assert.deepEqual(page.scatter(), [[1, 0.2, 3]]);
    page.unmount();
  });

  test(`Launchpad restores selected ${status} on reconnect without losing live monitoring`, async () => {
    const initial = { id: 'mining-A', status: 'running', progress: 25, miner: 'GP', config: 'test.json', logs: [] };
    const page = await downloadPage({ component: 'Launchpad', initialTasks: [initial] });
    page.sockets[0].open();
    await page.flush();
    page.selectTask(initial.id);
    page.emit({ type: 'log', task_id: initial.id, text: 'before reconnect' });
    page.emit({ type: 'scatter', task_id: initial.id, epoch: 1, ic: 0.1, complexity: 2 });
    page.sockets[0].close();
    await page.flush();
    page.runReconnectTimers();
    page.sockets.at(-1).open();
    // The API replays the existing task_update envelope; old running events cannot undo it.
    page.emit({ type: 'task_update', task: { ...initial, status, progress: 100 } });
    page.emit({ type: 'task_update', task: initial });
    page.emit({ type: 'log', task_id: 'other-task', text: 'unrelated task log' });
    page.emit({ type: 'log', task_id: initial.id, text: 'after reconnect' });
    page.emit({ type: 'scatter', task_id: initial.id, epoch: 2, ic: 0.2, complexity: 3 });
    await page.flush();
    assert.equal(page.connection().isConnected, true);
    assert.match(page.taskRow(initial.id), displayed);
    assert.match(page.screen(), new RegExp(`Status:\\s*${status}`));
    assert.match(page.screen(), /before reconnect/);
    assert.match(page.screen(), /after reconnect/);
    assert.doesNotMatch(page.screen(), /unrelated task log/);
    assert.deepEqual(page.scatter(), [[1, 0.1, 2], [2, 0.2, 3]]);
    page.unmount();
  });
}

test('Launchpad keeps new running progress over the initial snapshot and accepts later progress', async () => {
  const initial = { id: 'mining-A', status: 'running', progress: 25, miner: 'GP', config: 'test.json', logs: [] };
  const page = await downloadPage({ component: 'Launchpad', deferInitialTasks: true });
  page.emit({ type: 'task_update', task: { ...initial, progress: 60 } });
  await page.flush();
  await page.resolveTasks([initial]);
  assert.match(page.taskRow(initial.id), /运行中 \(\s*60\s*%\)/);
  page.emit({ type: 'task_update', task: { ...initial, progress: 80 } });
  await page.flush();
  assert.match(page.taskRow(initial.id), /运行中 \(\s*80\s*%\)/);
  page.unmount();
});

for (const [status, displayed] of [
  ['completed', /Successfully completed batch download!/],
  ['failed', /Batch download failed\. Review/],
  ['partial_failed', /Batch download partially failed/],
]) {
  test(`download reconnect restores the matching ${status} snapshot after a missed terminal`, async () => {
    const page = await downloadPage();
    page.sockets[0].open();
    await page.flush();
    page.click('Execute Batch Download');
    await page.accept();
    page.emit(event('running', 60));
    await page.flush();
    page.sockets[0].close();
    await page.flush();
    assert.match(page.screen(), /Monitoring connection interrupted/);
    assert.match(page.screen(), /60\s*%/);
    assert.doesNotMatch(page.screen(), /Batch download failed|Successfully completed batch/);
    page.runReconnectTimers();
    page.sockets.at(-1).open();
    // The server replays cached snapshots after reconnect, including unrelated jobs.
    page.emit(event('completed', 100, 'old-job'));
    page.emit(event(status));
    page.emit(event('running', 80));
    page.emit({ type: 'log', task_id: 'mining-A', text: 'unrelated log' });
    await page.flush();
    assert.match(page.screen(), displayed);
    assert.match(page.screen(), /Start another batch/);
    assert.doesNotMatch(page.screen(), /Monitoring connection interrupted/);
    if (status !== 'completed') assert.doesNotMatch(page.screen(), /Successfully completed batch/);
    page.unmount();
  });
}
