import { useCallback, useEffect, useMemo, useState } from 'react';
import { useSearchParams } from 'react-router-dom';
import ReactECharts from 'echarts-for-react';
import {
  BarChart3,
  CheckSquare,
  ChevronRight,
  CircleAlert,
  Database,
  FileSearch,
  GitBranch,
  GitCompareArrows,
  Loader2,
  RefreshCw,
  Save,
} from 'lucide-react';
import { useI18n } from '../i18n';

const API_BASE = 'http://localhost:8000';
const LIFECYCLE_STATUSES = ['DISCOVERED', 'INSPECTED', 'PAPER_TRADING', 'LIVE', 'RETIRED'];

type Metrics = Record<string, number>;

type FactorSummary = {
  factor_id: string;
  miner_type: string;
  lifecycle_status: string;
  logic_hash: string;
  metrics: Metrics;
  created_at: string;
  display: string;
  logic_kind: string;
  snapshot_available?: boolean;
};

type FactorDetail = {
  metadata: FactorSummary & {
    user_id: string;
    parent_ids: string[];
    generation_config: Record<string, unknown>;
    logic_reference: Record<string, unknown>;
  };
  logic: Record<string, any>;
  audit_snapshot: { values_available: boolean; message: string };
};

type AnalysisPoint = { timestamp: string; value: number };
type QuantilePoint = { bucket: string; mean_return: number; count: number };
type AnalysisPayload = {
  factor: FactorSummary;
  lineage: Record<string, unknown>;
  analysis: {
    mode: 'sequential_single' | 'cross_asset';
    rolling_ic: AnalysisPoint[];
    turnover: AnalysisPoint[];
    quantiles: QuantilePoint[];
    summary: {
      observations: number;
      start: string;
      end: string;
      rolling_window: number;
      latest_rolling_ic: number | null;
      mean_turnover: number | null;
      quantile_spread: number | null;
    };
  };
};

type TreeNode = { name: string; children?: TreeNode[] };

function metric(value?: number) {
  return typeof value === 'number' && Number.isFinite(value) ? value.toFixed(4) : '—';
}

function dateTime(value?: string) {
  if (!value) return '—';
  const date = new Date(value);
  return Number.isNaN(date.valueOf()) ? value : date.toLocaleString();
}

function lifecycleClass(status: string) {
  const classes: Record<string, string> = {
    DISCOVERED: 'border-slate-600 bg-slate-800/60 text-slate-300',
    INSPECTED: 'border-blue-800/70 bg-blue-950/40 text-blue-300',
    PAPER_TRADING: 'border-amber-800/70 bg-amber-950/30 text-amber-300',
    LIVE: 'border-emerald-800/70 bg-emerald-950/30 text-emerald-300',
    RETIRED: 'border-zinc-700 bg-zinc-900 text-zinc-400',
  };
  return classes[status] || classes.DISCOVERED;
}

function astToTree(node: any): TreeNode {
  if (!node || typeof node !== 'object' || !node.op) return { name: String(node ?? '∅') };
  const children = [node.left, node.right]
    .filter((child) => child !== undefined && child !== null)
    .map(astToTree);
  return { name: node.op, ...(children.length ? { children } : {}) };
}

function lineageValue(value: unknown) {
  if (Array.isArray(value)) return value.join(', ');
  if (typeof value === 'object' && value !== null) return JSON.stringify(value);
  return String(value ?? '—');
}

function TearsheetPanel({ payload }: { payload: AnalysisPayload }) {
  const { t } = useI18n();
  const { analysis, lineage } = payload;
  const lineOption = (title: string, points: AnalysisPoint[], color: string) => ({
    backgroundColor: 'transparent',
    tooltip: { trigger: 'axis' },
    grid: { top: 38, left: 44, right: 20, bottom: 38 },
    title: { text: title, left: 10, top: 8, textStyle: { color: '#e2e8f0', fontSize: 12, fontWeight: 600 } },
    xAxis: { type: 'category', data: points.map((point) => new Date(point.timestamp).toLocaleString()), axisLabel: { color: '#94a3b8', hideOverlap: true }, axisLine: { lineStyle: { color: '#334155' } } },
    yAxis: { type: 'value', axisLabel: { color: '#94a3b8', formatter: (value: number) => value.toFixed(3) }, splitLine: { lineStyle: { color: '#1e293b' } } },
    series: [{ type: 'line', data: points.map((point) => point.value), smooth: true, showSymbol: false, lineStyle: { color, width: 2 }, areaStyle: { opacity: 0.1 } }],
  });
  const quantileOption = {
    backgroundColor: 'transparent',
    tooltip: { trigger: 'axis', valueFormatter: (value: number) => value.toExponential(3) },
    grid: { top: 38, left: 50, right: 20, bottom: 28 },
    title: { text: t('inspector.quantileReturn'), left: 10, top: 8, textStyle: { color: '#e2e8f0', fontSize: 12, fontWeight: 600 } },
    xAxis: { type: 'category', data: analysis.quantiles.map((item) => item.bucket), axisLabel: { color: '#94a3b8' }, axisLine: { lineStyle: { color: '#334155' } } },
    yAxis: { type: 'value', axisLabel: { color: '#94a3b8', formatter: (value: number) => value.toExponential(1) }, splitLine: { lineStyle: { color: '#1e293b' } } },
    series: [{ type: 'bar', data: analysis.quantiles.map((item) => item.mean_return), itemStyle: { color: '#a78bfa', borderRadius: [3, 3, 0, 0] } }],
  };
  const summaryCards = [
    [t('inspector.observations'), String(analysis.summary.observations)],
    [t('inspector.latestRollingIc'), metric(analysis.summary.latest_rolling_ic ?? undefined)],
    [t('inspector.meanTurnover'), metric(analysis.summary.mean_turnover ?? undefined)],
    [t('inspector.quantileSpread'), metric(analysis.summary.quantile_spread ?? undefined)],
  ];
  const lineageEntries = Object.entries(lineage)
    .filter(([key]) => !['snapshot_file', 'snapshot_schema'].includes(key))
    .slice(0, 8);

  return (
    <div className="space-y-4">
      <div className="flex flex-wrap items-center justify-between gap-2 text-xs text-muted-foreground">
        <span>{t('inspector.realSnapshot')}</span>
        <span>{analysis.mode === 'cross_asset' ? t('inspector.crossAssetMethod') : t('inspector.sequentialMethod')}</span>
      </div>
      <div className="grid grid-cols-2 xl:grid-cols-4 gap-3">
        {summaryCards.map(([label, value]) => <div key={label} className="rounded-lg border border-border bg-secondary/20 p-3"><p className="text-[10px] uppercase tracking-wider text-muted-foreground">{label}</p><p className="mt-1 font-mono text-base font-semibold">{value}</p></div>)}
      </div>
      <div className="grid grid-cols-1 xl:grid-cols-2 gap-4">
        <div className="h-64 rounded-lg border border-border bg-background/30 overflow-hidden"><ReactECharts option={lineOption(t('inspector.rollingIc'), analysis.rolling_ic, '#38bdf8')} style={{ height: '100%', width: '100%' }} /></div>
        <div className="h-64 rounded-lg border border-border bg-background/30 overflow-hidden"><ReactECharts option={quantileOption} style={{ height: '100%', width: '100%' }} /></div>
      </div>
      <div className="h-56 rounded-lg border border-border bg-background/30 overflow-hidden"><ReactECharts option={lineOption(t('inspector.turnover'), analysis.turnover, '#f59e0b')} style={{ height: '100%', width: '100%' }} /></div>
      <div className="rounded-lg border border-border bg-secondary/10 p-4">
        <p className="text-xs font-bold uppercase tracking-wider text-muted-foreground">{t('inspector.dataLineage')}</p>
        <div className="mt-3 grid grid-cols-1 md:grid-cols-2 gap-x-5 gap-y-2 text-xs">
          {lineageEntries.map(([key, value]) => <div key={key} className="flex gap-2"><span className="shrink-0 text-muted-foreground">{key}</span><span className="font-mono break-all text-foreground">{lineageValue(value)}</span></div>)}
        </div>
      </div>
    </div>
  );
}

function ComparisonPanel({ payloads }: { payloads: AnalysisPayload[] }) {
  const { t } = useI18n();
  const timestamps = [...new Set(payloads.flatMap((payload) => payload.analysis.rolling_ic.map((point) => point.timestamp)))].sort();
  const option = {
    backgroundColor: 'transparent',
    tooltip: { trigger: 'axis' },
    legend: { top: 8, textStyle: { color: '#cbd5e1' } },
    grid: { top: 42, left: 44, right: 20, bottom: 38 },
    xAxis: { type: 'category', data: timestamps.map((timestamp) => new Date(timestamp).toLocaleString()), axisLabel: { color: '#94a3b8', hideOverlap: true }, axisLine: { lineStyle: { color: '#334155' } } },
    yAxis: { type: 'value', axisLabel: { color: '#94a3b8' }, splitLine: { lineStyle: { color: '#1e293b' } } },
    series: payloads.map((payload, index) => {
      const values = new Map(payload.analysis.rolling_ic.map((point) => [point.timestamp, point.value]));
      return { name: payload.factor.factor_id, type: 'line', smooth: true, showSymbol: false, data: timestamps.map((timestamp) => values.get(timestamp) ?? null), lineStyle: { width: 2 }, color: ['#38bdf8', '#a78bfa', '#34d399', '#fb7185', '#f59e0b'][index] };
    }),
  };
  return (
    <div className="rounded-xl border border-violet-900/60 bg-violet-950/10 p-5">
      <div className="flex items-center gap-2 mb-4"><GitCompareArrows className="w-4 h-4 text-violet-300" /><h3 className="text-sm font-bold">{t('inspector.comparison')}</h3><span className="ml-auto text-xs text-muted-foreground">{payloads.length} {t('inspector.factorsSelected')}</span></div>
      <div className="h-72"><ReactECharts option={option} style={{ height: '100%', width: '100%' }} /></div>
    </div>
  );
}

function LogicPanel({ detail }: { detail: FactorDetail }) {
  const { t } = useI18n();
  const logic = detail.logic;
  const treeOption = useMemo(() => ({
    backgroundColor: 'transparent',
    tooltip: { trigger: 'item', triggerOn: 'mousemove' },
    series: [{
      type: 'tree',
      data: [astToTree(logic.ast)],
      top: '8%', left: '8%', bottom: '8%', right: '8%',
      symbolSize: 9,
      itemStyle: { color: '#3b82f6', borderColor: '#93c5fd' },
      lineStyle: { color: '#475569', width: 1.5, curveness: 0.4 },
      label: {
        position: 'top', verticalAlign: 'middle', align: 'center',
        fontSize: 11, color: '#e2e8f0', backgroundColor: '#172033',
        padding: [3, 7], borderRadius: 4, borderColor: '#334155', borderWidth: 1,
      },
      leaves: { label: { position: 'bottom', verticalAlign: 'middle', align: 'center' } },
      expandAndCollapse: true,
      animationDuration: 350,
    }],
  }), [logic.ast]);

  if (logic.kind === 'ast') {
    return (
      <div className="grid grid-cols-1 xl:grid-cols-2 gap-4 min-h-[310px]">
        <div className="min-h-[280px] rounded-lg border border-border bg-background/40 overflow-hidden">
          <ReactECharts option={treeOption} style={{ height: '100%', minHeight: 280, width: '100%' }} />
        </div>
        <pre className="rounded-lg border border-border bg-black/50 p-4 overflow-auto font-mono text-xs leading-6 text-emerald-300 whitespace-pre-wrap break-words">
          {logic.display}
        </pre>
      </div>
    );
  }

  if (logic.kind === 'source') {
    return (
      <div className="space-y-3">
        <div className="text-xs text-muted-foreground">{t('inspector.source')}: <span className="font-mono text-foreground">{logic.source_file || t('inspector.missing')}</span></div>
        <pre className="min-h-[260px] rounded-lg border border-border bg-black/50 p-4 overflow-auto font-mono text-xs leading-6 text-emerald-300 whitespace-pre-wrap">
          {logic.source || t('inspector.sourceMissing')}
        </pre>
        {logic.reflection && <div className="rounded-lg border border-border bg-secondary/20 p-3 text-xs text-muted-foreground whitespace-pre-wrap">{logic.reflection}</div>}
      </div>
    );
  }

  if (logic.kind === 'actions') {
    return (
      <div className="rounded-lg border border-border bg-black/50 p-4 min-h-[180px]">
        <div className="flex flex-wrap items-center gap-2">
          {(logic.actions || []).length ? logic.actions.map((action: string, index: number) => (
            <div key={`${action}-${index}`} className="flex items-center gap-2">
              <span className="rounded bg-secondary px-3 py-2 font-mono text-xs text-foreground">{action}</span>
              {index < logic.actions.length - 1 && <ChevronRight className="w-3 h-3 text-muted-foreground" />}
            </div>
          )) : <span className="text-sm text-muted-foreground">{t('inspector.noActions')}</span>}
        </div>
        {logic.weights_file && <p className="mt-5 text-xs text-muted-foreground">{t('inspector.agentArtifact')}: <span className="font-mono">{logic.weights_file}</span></p>}
      </div>
    );
  }

  if (logic.kind === 'nn_channel') {
    return (
      <div className="grid grid-cols-1 sm:grid-cols-3 gap-4">
        <div className="rounded-lg border border-border bg-secondary/20 p-4">
          <p className="text-xs text-muted-foreground">{t('inspector.modelVersion')}</p>
          <p className="mt-2 font-mono text-sm break-all">{logic.model_version || '—'}</p>
        </div>
        <div className="rounded-lg border border-border bg-secondary/20 p-4">
          <p className="text-xs text-muted-foreground">{t('inspector.outputChannel')}</p>
          <p className="mt-2 text-2xl font-semibold">{logic.channel ?? '—'}</p>
        </div>
        <div className="rounded-lg border border-border bg-secondary/20 p-4">
          <p className="text-xs text-muted-foreground">{t('inspector.weightsArtifact')}</p>
          <p className={`mt-2 text-sm font-medium ${logic.weights_available ? 'text-emerald-400' : 'text-amber-400'}`}>
            {logic.weights_available ? t('inspector.available') : t('inspector.missing')}
          </p>
          <p className="mt-1 font-mono text-[11px] text-muted-foreground break-all">{logic.weights_file || '—'}</p>
        </div>
        <p className="sm:col-span-3 text-xs text-muted-foreground">
          {t('inspector.nnPhaseOne')}
        </p>
      </div>
    );
  }

  return <pre className="rounded-lg border border-border bg-black/50 p-4 text-xs text-muted-foreground overflow-auto">{JSON.stringify(logic.reference || {}, null, 2)}</pre>;
}

export function Inspector() {
  const { t } = useI18n();
  const [searchParams, setSearchParams] = useSearchParams();
  const [factors, setFactors] = useState<FactorSummary[]>([]);
  const [selectedId, setSelectedId] = useState(searchParams.get('factor') || '');
  const [detail, setDetail] = useState<FactorDetail | null>(null);
  const [query, setQuery] = useState('');
  const [minerFilter, setMinerFilter] = useState('ALL');
  const [lifecycleFilter, setLifecycleFilter] = useState('ALL');
  const [sortBy, setSortBy] = useState('created_at');
  const [statusDraft, setStatusDraft] = useState('DISCOVERED');
  const [selectedForReview, setSelectedForReview] = useState<string[]>([]);
  const [batchStatus, setBatchStatus] = useState('INSPECTED');
  const [analysis, setAnalysis] = useState<AnalysisPayload | null>(null);
  const [comparison, setComparison] = useState<AnalysisPayload[]>([]);
  const [loadingCatalog, setLoadingCatalog] = useState(true);
  const [loadingDetail, setLoadingDetail] = useState(false);
  const [loadingAnalysis, setLoadingAnalysis] = useState(false);
  const [savingBatch, setSavingBatch] = useState(false);
  const [loadingComparison, setLoadingComparison] = useState(false);
  const [savingStatus, setSavingStatus] = useState(false);
  const [error, setError] = useState('');

  const loadCatalog = useCallback(async () => {
    setLoadingCatalog(true);
    try {
      const response = await fetch(`${API_BASE}/api/factors?limit=500`);
      if (!response.ok) throw new Error('Unable to load the factor catalog.');
      const payload = await response.json();
      setFactors(payload.factors || []);
      setError('');
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Unable to load the factor catalog.');
    } finally {
      setLoadingCatalog(false);
    }
  }, []);

  useEffect(() => { void loadCatalog(); }, [loadCatalog]);

  useEffect(() => {
    const factorFromUrl = searchParams.get('factor') || '';
    setSelectedId(factorFromUrl);
  }, [searchParams]);

  useEffect(() => {
    if (!selectedId) {
      setDetail(null);
      return;
    }
    const loadDetail = async () => {
      setLoadingDetail(true);
      try {
        const response = await fetch(`${API_BASE}/api/factors/${encodeURIComponent(selectedId)}`);
        if (!response.ok) throw new Error('This factor no longer exists in the local catalog.');
        const payload = await response.json();
        setDetail(payload);
        setStatusDraft(payload.metadata.lifecycle_status);
        setError('');
      } catch (err) {
        setDetail(null);
        setError(err instanceof Error ? err.message : 'Unable to load factor details.');
      } finally {
        setLoadingDetail(false);
      }
    };
    void loadDetail();
  }, [selectedId]);

  useEffect(() => {
    if (!selectedId) {
      setAnalysis(null);
      return;
    }
    const loadAnalysis = async () => {
      setLoadingAnalysis(true);
      try {
        const response = await fetch(`${API_BASE}/api/factors/${encodeURIComponent(selectedId)}/analysis`);
        if (response.status === 409) {
          setAnalysis(null);
          return;
        }
        if (!response.ok) throw new Error('Unable to calculate the persisted Tearsheet.');
        setAnalysis(await response.json());
      } catch (err) {
        setAnalysis(null);
        setError(err instanceof Error ? err.message : 'Unable to calculate the persisted Tearsheet.');
      } finally {
        setLoadingAnalysis(false);
      }
    };
    void loadAnalysis();
  }, [selectedId]);

  const filteredFactors = useMemo(() => factors
    .filter((factor) => minerFilter === 'ALL' || factor.miner_type === minerFilter)
    .filter((factor) => lifecycleFilter === 'ALL' || factor.lifecycle_status === lifecycleFilter)
    .filter((factor) => {
      const searchable = `${factor.factor_id} ${factor.miner_type} ${factor.logic_hash} ${factor.display}`.toLowerCase();
      return searchable.includes(query.toLowerCase());
    })
    .sort((a, b) => {
      if (sortBy === 'fitness') return (b.metrics.fitness_score ?? -Infinity) - (a.metrics.fitness_score ?? -Infinity);
      if (sortBy === 'ic') return (b.metrics.IC ?? -Infinity) - (a.metrics.IC ?? -Infinity);
      return (b.created_at || '').localeCompare(a.created_at || '');
    }), [factors, lifecycleFilter, minerFilter, query, sortBy]);

  const minerTypes = useMemo(() => [...new Set(factors.map((factor) => factor.miner_type))].sort(), [factors]);

  const selectFactor = (factorId: string) => setSearchParams({ factor: factorId });

  const toggleReviewSelection = (factorId: string) => {
    setSelectedForReview((current) => current.includes(factorId)
      ? current.filter((id) => id !== factorId)
      : [...current, factorId]);
  };

  const saveLifecycle = async () => {
    if (!detail) return;
    setSavingStatus(true);
    try {
      const response = await fetch(`${API_BASE}/api/factors/${encodeURIComponent(detail.metadata.factor_id)}/lifecycle`, {
        method: 'PATCH',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ lifecycle_status: statusDraft }),
      });
      if (!response.ok) throw new Error('Unable to update lifecycle status.');
      const payload = await response.json();
      setDetail(payload);
      setFactors((current) => current.map((factor) => factor.factor_id === payload.metadata.factor_id
        ? { ...factor, lifecycle_status: payload.metadata.lifecycle_status }
        : factor));
      setError('');
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Unable to update lifecycle status.');
    } finally {
      setSavingStatus(false);
    }
  };

  const saveBatchLifecycle = async () => {
    if (!selectedForReview.length) return;
    setSavingBatch(true);
    try {
      const response = await fetch(`${API_BASE}/api/factors/lifecycle/batch`, {
        method: 'PATCH',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ factor_ids: selectedForReview, lifecycle_status: batchStatus }),
      });
      if (!response.ok) throw new Error('Unable to update selected factor lifecycles.');
      const payload = await response.json();
      const updated = new Map((payload.updated || []).map((factor: FactorSummary) => [factor.factor_id, factor.lifecycle_status]));
      setFactors((current) => current.map((factor) => updated.has(factor.factor_id)
        ? { ...factor, lifecycle_status: updated.get(factor.factor_id) as string }
        : factor));
      if (detail && updated.has(detail.metadata.factor_id)) {
        setDetail({ ...detail, metadata: { ...detail.metadata, lifecycle_status: updated.get(detail.metadata.factor_id) as string } });
        setStatusDraft(updated.get(detail.metadata.factor_id) as string);
      }
      setSelectedForReview([]);
      setError('');
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Unable to update selected factor lifecycles.');
    } finally {
      setSavingBatch(false);
    }
  };

  const compareSelected = async () => {
    if (selectedForReview.length < 2 || selectedForReview.length > 5) return;
    setLoadingComparison(true);
    try {
      const response = await fetch(`${API_BASE}/api/factors/compare`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ factor_ids: selectedForReview }),
      });
      if (!response.ok) throw new Error('Selected factors require persisted analysis snapshots before comparison.');
      const payload = await response.json();
      setComparison(payload.factors || []);
      setError('');
    } catch (err) {
      setComparison([]);
      setError(err instanceof Error ? err.message : 'Unable to compare selected factors.');
    } finally {
      setLoadingComparison(false);
    }
  };

  return (
    <div className="flex flex-col gap-6 max-w-[1500px] mx-auto min-h-[720px]">
      <header className="flex flex-col gap-4 xl:flex-row xl:items-end xl:justify-between">
        <div>
          <div className="flex items-center gap-2 text-primary">
            <FileSearch className="w-5 h-5" />
            <span className="text-xs font-bold tracking-[0.18em] uppercase">{t('inspector.kicker')}</span>
          </div>
          <h1 className="mt-2 text-3xl font-bold tracking-tight">{t('inspector.title')}</h1>
          <p className="mt-1 text-sm text-muted-foreground">{t('inspector.subtitle')}</p>
        </div>
        <button onClick={() => void loadCatalog()} className="inline-flex items-center justify-center gap-2 rounded-md border border-border bg-card px-3 py-2 text-sm hover:bg-secondary transition-colors" disabled={loadingCatalog}>
          <RefreshCw className={`w-4 h-4 ${loadingCatalog ? 'animate-spin' : ''}`} /> {t('inspector.refreshCatalog')}
        </button>
      </header>

      {error && <div className="flex items-center gap-2 rounded-lg border border-red-900/60 bg-red-950/25 px-4 py-3 text-sm text-red-300"><CircleAlert className="w-4 h-4 shrink-0" />{error}</div>}

      {selectedForReview.length > 0 && (
        <div className="flex flex-col gap-3 rounded-xl border border-violet-900/60 bg-violet-950/15 p-4 lg:flex-row lg:items-center lg:justify-between">
          <div className="flex items-center gap-2 text-sm"><CheckSquare className="w-4 h-4 text-violet-300" /><span>{selectedForReview.length} {t('inspector.factorsSelected')}</span></div>
          <div className="flex flex-wrap items-center gap-2">
            <select value={batchStatus} onChange={(event) => setBatchStatus(event.target.value)} className="rounded-md border border-input bg-background px-2 py-2 text-xs">
              {LIFECYCLE_STATUSES.map((status) => <option key={status} value={status}>{t(`status.${status}`)}</option>)}
            </select>
            <button onClick={() => void saveBatchLifecycle()} disabled={savingBatch} className="inline-flex items-center gap-2 rounded-md border border-violet-700/60 bg-violet-950/40 px-3 py-2 text-xs font-bold text-violet-200 disabled:opacity-50"><Save className="w-3.5 h-3.5" />{t('inspector.batchReview')}</button>
            <button onClick={() => void compareSelected()} disabled={loadingComparison || selectedForReview.length < 2 || selectedForReview.length > 5} className="inline-flex items-center gap-2 rounded-md bg-primary px-3 py-2 text-xs font-bold text-primary-foreground disabled:opacity-50"><GitCompareArrows className="w-3.5 h-3.5" />{loadingComparison ? t('common.loading') : t('inspector.compare')}</button>
          </div>
        </div>
      )}

      <div className="grid grid-cols-1 xl:grid-cols-3 gap-6 flex-1">
        <aside className="xl:col-span-1 border border-border bg-card rounded-xl overflow-hidden flex flex-col min-h-[580px]">
          <div className="p-4 border-b border-border space-y-3">
            <div className="flex items-center justify-between">
              <h2 className="font-semibold text-sm">{t('inspector.catalog')}</h2>
              <span className="rounded bg-secondary px-2 py-0.5 text-xs text-muted-foreground">{filteredFactors.length} / {factors.length}</span>
            </div>
            <input value={query} onChange={(event) => setQuery(event.target.value)} placeholder={t('inspector.search')} className="w-full rounded-md border border-input bg-background px-3 py-2 text-sm outline-none focus:border-primary" />
            <div className="grid grid-cols-3 gap-2">
              <select value={minerFilter} onChange={(event) => setMinerFilter(event.target.value)} className="min-w-0 rounded-md border border-input bg-background px-2 py-2 text-xs">
                <option value="ALL">{t('inspector.allMiners')}</option>
                {minerTypes.map((miner) => <option key={miner} value={miner}>{miner}</option>)}
              </select>
              <select value={lifecycleFilter} onChange={(event) => setLifecycleFilter(event.target.value)} className="min-w-0 rounded-md border border-input bg-background px-2 py-2 text-xs">
                <option value="ALL">{t('inspector.allStates')}</option>
                {LIFECYCLE_STATUSES.map((status) => <option key={status} value={status}>{t(`status.${status}`)}</option>)}
              </select>
              <select value={sortBy} onChange={(event) => setSortBy(event.target.value)} className="min-w-0 rounded-md border border-input bg-background px-2 py-2 text-xs">
                <option value="created_at">{t('inspector.newest')}</option>
                <option value="fitness">{t('inspector.fitness')}</option>
                <option value="ic">IC</option>
              </select>
            </div>
          </div>

          <div className="flex-1 overflow-y-auto divide-y divide-border">
            {loadingCatalog && <div className="p-8 flex justify-center"><Loader2 className="w-5 h-5 animate-spin text-primary" /></div>}
            {!loadingCatalog && filteredFactors.length === 0 && <div className="p-8 text-center text-sm text-muted-foreground">{t('inspector.noMatches')}</div>}
            {filteredFactors.map((factor) => (
              <div key={factor.factor_id} className={`flex gap-2 p-3 transition-colors hover:bg-secondary/50 ${selectedId === factor.factor_id ? 'bg-primary/10 border-l-2 border-primary' : 'border-l-2 border-transparent'}`}>
                <input aria-label={t('inspector.selectForReview')} type="checkbox" checked={selectedForReview.includes(factor.factor_id)} onChange={() => toggleReviewSelection(factor.factor_id)} className="mt-1 h-3.5 w-3.5 accent-primary" />
                <button onClick={() => selectFactor(factor.factor_id)} className="min-w-0 flex-1 text-left">
                  <div className="flex gap-2 items-start justify-between">
                    <span className="font-mono text-xs font-semibold text-foreground">{factor.factor_id}</span>
                    <span className={`shrink-0 rounded border px-1.5 py-0.5 text-[10px] font-semibold ${lifecycleClass(factor.lifecycle_status)}`}>{t(`status.${factor.lifecycle_status}`)}</span>
                  </div>
                  <div className="mt-2 flex items-center gap-2 text-xs text-muted-foreground"><span className="rounded bg-secondary px-1.5 py-0.5">{factor.miner_type}</span><span>IC {metric(factor.metrics.IC)}</span><span>Fit {metric(factor.metrics.fitness_score)}</span></div>
                  <p className="mt-2 font-mono text-xs leading-5 text-muted-foreground line-clamp-2">{factor.display}</p>
                  <div className="mt-2 flex justify-between gap-2 text-[11px] text-muted-foreground/70"><span>{dateTime(factor.created_at)}</span>{factor.snapshot_available && <span className="text-emerald-400">{t('inspector.snapshotReady')}</span>}</div>
                </button>
              </div>
            ))}
          </div>
        </aside>

        <section className="xl:col-span-2 min-w-0">
          {!selectedId && <div className="h-full min-h-[580px] rounded-xl border border-dashed border-border bg-card/40 flex flex-col items-center justify-center text-center p-8"><Database className="w-10 h-10 text-muted-foreground mb-4" /><h2 className="font-semibold">{t('inspector.selectFactor')}</h2><p className="mt-2 max-w-md text-sm text-muted-foreground">{t('inspector.selectHint')}</p></div>}
          {selectedId && loadingDetail && <div className="h-full min-h-[580px] rounded-xl border border-border bg-card flex items-center justify-center"><Loader2 className="w-6 h-6 animate-spin text-primary" /></div>}
          {selectedId && !loadingDetail && detail && (
            <div className="space-y-6">
              {comparison.length > 0 && <ComparisonPanel payloads={comparison} />}

              <div className="rounded-xl border border-border bg-card p-5">
                <div className="flex flex-col gap-4 lg:flex-row lg:items-start lg:justify-between">
                  <div className="min-w-0">
                    <div className="flex flex-wrap items-center gap-2">
                      <h2 className="font-mono text-lg font-bold">{detail.metadata.factor_id}</h2>
                      <span className={`rounded border px-2 py-1 text-[11px] font-bold ${lifecycleClass(detail.metadata.lifecycle_status)}`}>{t(`status.${detail.metadata.lifecycle_status}`)}</span>
                      <span className="rounded border border-purple-900/70 bg-purple-950/30 px-2 py-1 text-[11px] font-bold text-purple-300">{detail.metadata.miner_type}</span>
                    </div>
                    <p className="mt-2 font-mono text-xs text-muted-foreground break-all">{t('inspector.logicHash')}: {detail.metadata.logic_hash || t('inspector.missing')}</p>
                  </div>
                  <div className="flex gap-2 items-center">
                    <select value={statusDraft} onChange={(event) => setStatusDraft(event.target.value)} className="rounded-md border border-input bg-background px-2 py-2 text-xs">
                      {LIFECYCLE_STATUSES.map((status) => <option key={status} value={status}>{t(`status.${status}`)}</option>)}
                    </select>
                    <button onClick={() => void saveLifecycle()} disabled={savingStatus || statusDraft === detail.metadata.lifecycle_status} className="inline-flex items-center gap-2 rounded-md bg-primary px-3 py-2 text-xs font-bold text-primary-foreground disabled:opacity-50"><Save className="w-3.5 h-3.5" />{savingStatus ? t('inspector.saving') : t('inspector.saveStatus')}</button>
                  </div>
                </div>
                <div className="mt-5 grid grid-cols-2 lg:grid-cols-4 gap-3">
                  {[
                    [t('inspector.fitness'), metric(detail.metadata.metrics.fitness_score)],
                    ['IC', metric(detail.metadata.metrics.IC)],
                    [t('inspector.rankIc'), metric(detail.metadata.metrics.RankIC)],
                    [t('inspector.turnover'), metric(detail.metadata.metrics.Turnover)],
                  ].map(([label, value]) => <div key={label} className="rounded-lg border border-border bg-secondary/20 p-3"><p className="text-[11px] uppercase tracking-wider text-muted-foreground">{label}</p><p className="mt-1 font-mono text-lg font-semibold">{value}</p></div>)}
                </div>
                <div className="mt-4 grid grid-cols-1 sm:grid-cols-3 gap-3 text-xs">
                  <div><span className="text-muted-foreground">{t('inspector.created')}</span><p className="mt-1">{dateTime(detail.metadata.created_at)}</p></div>
                  <div><span className="text-muted-foreground">{t('inspector.researcher')}</span><p className="mt-1">{detail.metadata.user_id || '—'}</p></div>
                  <div><span className="text-muted-foreground">{t('inspector.lineage')}</span><p className="mt-1 font-mono break-all">{detail.metadata.parent_ids?.length ? detail.metadata.parent_ids.join(', ') : t('inspector.noneRecorded')}</p></div>
                </div>
              </div>

              <div className="rounded-xl border border-border bg-card p-5">
                <div className="flex items-center gap-2 mb-4"><GitBranch className="w-4 h-4 text-primary" /><h3 className="text-sm font-bold">{t('inspector.logicReference')}</h3><span className="ml-auto text-xs text-muted-foreground">{detail.logic.kind}</span></div>
                <LogicPanel detail={detail} />
              </div>

              <div className={`rounded-xl border p-5 ${detail.audit_snapshot.values_available ? 'border-emerald-900/60 bg-emerald-950/10' : 'border-amber-900/60 bg-amber-950/10'}`}>
                <div className="flex gap-3"><BarChart3 className={`w-5 h-5 shrink-0 ${detail.audit_snapshot.values_available ? 'text-emerald-400' : 'text-amber-400'}`} /><div><h3 className="text-sm font-bold">{t('inspector.tearsheet')}</h3><p className="mt-1 text-sm text-muted-foreground">{detail.audit_snapshot.message}</p></div></div>
                {loadingAnalysis && <div className="flex justify-center py-10"><Loader2 className="w-5 h-5 animate-spin text-primary" /></div>}
                {!loadingAnalysis && analysis && <div className="mt-5"><TearsheetPanel payload={analysis} /></div>}
                {!loadingAnalysis && !analysis && <p className="mt-3 text-xs text-muted-foreground">{t('inspector.snapshotRequired')}</p>}
              </div>
            </div>
          )}
        </section>
      </div>
    </div>
  );
}
