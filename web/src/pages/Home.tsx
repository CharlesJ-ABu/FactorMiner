import { useEffect, useMemo, useState } from 'react';
import { Link } from 'react-router-dom';
import {
  Activity,
  AlertTriangle,
  ArrowRight,
  CheckCircle2,
  Clock3,
  Database,
  FileSearch,
  FlaskConical,
  Gauge,
  Loader2,
  Play,
  Rocket,
  ShieldCheck,
  Target,
  XCircle,
} from 'lucide-react';
import { useI18n } from '../i18n';

const API_BASE = 'http://localhost:8000';

type Factor = {
  factor_id: string;
  miner_type: string;
  lifecycle_status: string;
  metrics: Record<string, number>;
  display: string;
};

type Task = {
  id: string;
  status: string;
  miner: string;
  config: string;
  progress: number;
  duration: string;
  start_time: string;
  result_count?: number;
};

type Dashboard = {
  engine_online: boolean;
  generated_at: string;
  tasks: { total: number; statuses: Record<string, number>; success_rate: number | null; recent: Task[] };
  factors: {
    total: number;
    reviewed: number;
    by_miner: Record<string, number>;
    by_lifecycle: Record<string, number>;
    top_by_fitness: Factor[];
    top_by_ic: Factor | null;
  };
};

const emptyDashboard: Dashboard = {
  engine_online: false,
  generated_at: '',
  tasks: { total: 0, statuses: {}, success_rate: null, recent: [] },
  factors: { total: 0, reviewed: 0, by_miner: {}, by_lifecycle: {}, top_by_fitness: [], top_by_ic: null },
};

const minerColors = ['bg-cyan-400', 'bg-violet-400', 'bg-amber-400', 'bg-emerald-400', 'bg-rose-400'];

function metric(value?: number) {
  return typeof value === 'number' && Number.isFinite(value) ? value.toFixed(4) : '—';
}

function time(value?: string) {
  if (!value) return '—';
  const date = new Date(value);
  return Number.isNaN(date.valueOf()) ? value : date.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });
}

function statusClass(status: string) {
  const styles: Record<string, string> = {
    running: 'text-cyan-300 border-cyan-800/70 bg-cyan-950/30',
    completed: 'text-emerald-300 border-emerald-800/70 bg-emerald-950/30',
    completed_empty: 'text-amber-300 border-amber-800/70 bg-amber-950/30',
    failed: 'text-red-300 border-red-800/70 bg-red-950/30',
  };
  return styles[status] || styles.failed;
}

export function Home() {
  const { t } = useI18n();
  const [dashboard, setDashboard] = useState<Dashboard>(emptyDashboard);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState('');

  useEffect(() => {
    let active = true;
    const loadDashboard = async () => {
      try {
        const response = await fetch(`${API_BASE}/api/dashboard`);
        if (!response.ok) throw new Error('Dashboard data is unavailable.');
        const data = await response.json();
        if (!active) return;
        setDashboard(data);
        setError('');
      } catch (err) {
        if (!active) return;
        setError(err instanceof Error ? err.message : 'Dashboard data is unavailable.');
      } finally {
        if (active) setLoading(false);
      }
    };
    void loadDashboard();
    const interval = window.setInterval(() => void loadDashboard(), 5000);
    return () => { active = false; window.clearInterval(interval); };
  }, []);

  const reviewCoverage = dashboard.factors.total
    ? Math.round((dashboard.factors.reviewed / dashboard.factors.total) * 100)
    : 0;
  const successRate = dashboard.tasks.success_rate === null ? '—' : `${(dashboard.tasks.success_rate * 100).toFixed(1)}%`;
  const minerMix = useMemo(() => Object.entries(dashboard.factors.by_miner), [dashboard.factors.by_miner]);

  return (
    <div className="max-w-[1500px] mx-auto pb-10 space-y-6">
      <section className="relative overflow-hidden rounded-2xl border border-sky-900/60 bg-gradient-to-br from-slate-950 via-slate-950 to-sky-950/50 p-6 md:p-8">
        <div className="absolute -right-20 -top-24 h-72 w-72 rounded-full bg-sky-500/10 blur-3xl" />
        <div className="absolute right-36 -bottom-24 h-64 w-64 rounded-full bg-violet-500/10 blur-3xl" />
        <div className="relative flex flex-col gap-6 xl:flex-row xl:items-end xl:justify-between">
          <div className="max-w-2xl">
            <div className="flex items-center gap-2 text-sky-300"><Gauge className="w-4 h-4" /><span className="text-xs font-bold uppercase tracking-[0.2em]">{t('home.kicker')}</span></div>
            <h1 className="mt-3 text-3xl md:text-4xl font-bold tracking-tight">{t('home.title')}</h1>
            <p className="mt-3 text-sm leading-6 text-slate-300">{t('home.subtitle')}</p>
          </div>
          <div className="flex flex-wrap gap-3">
            <Link to="/launchpad" className="inline-flex items-center gap-2 rounded-md bg-sky-400 px-4 py-2.5 text-sm font-bold text-slate-950 hover:bg-sky-300 transition-colors"><Play className="w-4 h-4 fill-current" />{t('home.startMining')}</Link>
            <Link to="/inspector" className="inline-flex items-center gap-2 rounded-md border border-slate-700 bg-slate-900/80 px-4 py-2.5 text-sm font-semibold text-slate-100 hover:bg-slate-800 transition-colors"><FileSearch className="w-4 h-4" />{t('home.reviewArchive')}</Link>
          </div>
        </div>
        <div className="relative mt-7 flex flex-wrap items-center gap-3 text-xs">
          <span className={`inline-flex items-center gap-2 rounded-full border px-3 py-1.5 ${dashboard.engine_online ? 'border-emerald-800/70 bg-emerald-950/30 text-emerald-300' : 'border-red-800/70 bg-red-950/30 text-red-300'}`}><span className={`h-2 w-2 rounded-full ${dashboard.engine_online ? 'bg-emerald-400 animate-pulse' : 'bg-red-400'}`} />{dashboard.engine_online ? t('engine.online') : t('engine.offline')}</span>
          <span className="text-slate-400">{loading ? t('home.syncing') : t('home.snapshot', { time: time(dashboard.generated_at) })}</span>
        </div>
      </section>

      {error && <div className="rounded-lg border border-red-900/60 bg-red-950/20 px-4 py-3 text-sm text-red-300">{error}</div>}

      <section className="grid grid-cols-1 sm:grid-cols-2 xl:grid-cols-4 gap-4">
        {[
          { label: t('home.factorArchive'), value: dashboard.factors.total, note: t('home.factorArchiveNote'), icon: Database, color: 'text-sky-300 bg-sky-950/35 border-sky-900/60' },
          { label: t('home.reviewCoverage'), value: `${reviewCoverage}%`, note: t('home.reviewNote', { count: dashboard.factors.reviewed }), icon: ShieldCheck, color: 'text-violet-300 bg-violet-950/35 border-violet-900/60' },
          { label: t('home.bestIc'), value: metric(dashboard.factors.top_by_ic?.metrics.IC), note: dashboard.factors.top_by_ic ? dashboard.factors.top_by_ic.factor_id : t('home.noFactors'), icon: Target, color: 'text-emerald-300 bg-emerald-950/35 border-emerald-900/60' },
          { label: t('home.taskCompletion'), value: successRate, note: t('home.runningNote', { count: dashboard.tasks.statuses.running || 0 }), icon: Activity, color: 'text-amber-300 bg-amber-950/35 border-amber-900/60' },
        ].map((card) => {
          const Icon = card.icon;
          return <div key={card.label} className={`rounded-xl border p-5 ${card.color}`}><div className="flex items-center justify-between"><p className="text-xs font-semibold uppercase tracking-wider opacity-80">{card.label}</p><Icon className="w-4 h-4" /></div><p className="mt-4 text-3xl font-bold tracking-tight">{card.value}</p><p className="mt-2 text-xs opacity-75 truncate">{card.note}</p></div>;
        })}
      </section>

      <section className="grid grid-cols-1 xl:grid-cols-5 gap-6">
        <div className="xl:col-span-3 rounded-xl border border-border bg-card overflow-hidden">
          <div className="flex items-center justify-between border-b border-border bg-secondary/20 px-5 py-4"><div><p className="text-xs font-bold uppercase tracking-wider text-muted-foreground">{t('home.researchPulse')}</p><h2 className="mt-1 font-semibold">{t('home.highestFitness')}</h2></div><Link to="/inspector" className="text-xs text-primary hover:underline inline-flex items-center gap-1">{t('home.openInspector')} <ArrowRight className="w-3 h-3" /></Link></div>
          <div className="overflow-x-auto">
            <table className="w-full text-left text-sm"><thead className="bg-background/60 text-[11px] uppercase tracking-wider text-muted-foreground"><tr><th className="px-5 py-3 font-medium">{t('home.factor')}</th><th className="px-3 py-3 font-medium">{t('home.miner')}</th><th className="px-3 py-3 text-right font-medium">{t('home.fitness')}</th><th className="px-5 py-3 text-right font-medium">{t('home.ic')}</th></tr></thead><tbody className="divide-y divide-border">
              {dashboard.factors.top_by_fitness.map((factor) => <tr key={factor.factor_id} className="hover:bg-secondary/30"><td className="px-5 py-3"><Link to={`/inspector?factor=${encodeURIComponent(factor.factor_id)}`} className="font-mono text-xs text-primary hover:underline">{factor.factor_id}</Link><p className="mt-1 max-w-[330px] truncate font-mono text-[11px] text-muted-foreground">{factor.display}</p></td><td className="px-3 py-3"><span className="rounded bg-secondary px-2 py-1 text-xs">{factor.miner_type}</span></td><td className="px-3 py-3 text-right font-mono">{metric(factor.metrics.fitness_score)}</td><td className="px-5 py-3 text-right font-mono">{metric(factor.metrics.IC)}</td></tr>)}
              {!loading && dashboard.factors.top_by_fitness.length === 0 && <tr><td colSpan={4} className="px-5 py-10 text-center text-sm text-muted-foreground">{t('home.noArchivedFactors')}</td></tr>}
              {loading && <tr><td colSpan={4} className="px-5 py-10 text-center"><Loader2 className="inline w-5 h-5 animate-spin text-primary" /></td></tr>}
            </tbody></table>
          </div>
        </div>

        <div className="xl:col-span-2 rounded-xl border border-border bg-card p-5">
          <div className="flex items-center gap-2"><FlaskConical className="w-4 h-4 text-primary" /><div><p className="text-xs font-bold uppercase tracking-wider text-muted-foreground">{t('home.archiveComposition')}</p><h2 className="mt-1 font-semibold">{t('home.minerDistribution')}</h2></div></div>
          <div className="mt-7 h-3 overflow-hidden rounded-full bg-secondary flex">
            {minerMix.map(([miner, count], index) => <div key={miner} className={minerColors[index % minerColors.length]} style={{ width: `${dashboard.factors.total ? (count / dashboard.factors.total) * 100 : 0}%` }} title={`${miner}: ${count}`} />)}
          </div>
          <div className="mt-5 space-y-3">
            {minerMix.map(([miner, count], index) => <div key={miner} className="flex items-center justify-between text-sm"><span className="flex items-center gap-2"><span className={`w-2.5 h-2.5 rounded-full ${minerColors[index % minerColors.length]}`} />{miner}</span><span className="font-mono text-muted-foreground">{count} · {dashboard.factors.total ? Math.round((count / dashboard.factors.total) * 100) : 0}%</span></div>)}
            {!loading && minerMix.length === 0 && <p className="text-sm text-muted-foreground">{t('home.archiveEmpty')}</p>}
          </div>
          <div className="mt-7 grid grid-cols-2 gap-3 border-t border-border pt-5"><div><p className="text-xs text-muted-foreground">{t('home.discovered')}</p><p className="mt-1 text-xl font-semibold">{dashboard.factors.by_lifecycle.DISCOVERED || 0}</p></div><div><p className="text-xs text-muted-foreground">{t('home.reviewedLive')}</p><p className="mt-1 text-xl font-semibold">{dashboard.factors.reviewed}</p></div></div>
        </div>
      </section>

      <section className="grid grid-cols-1 xl:grid-cols-5 gap-6">
        <div className="xl:col-span-2 rounded-xl border border-border bg-card p-5">
          <div className="flex items-center gap-2"><Rocket className="w-4 h-4 text-primary" /><div><p className="text-xs font-bold uppercase tracking-wider text-muted-foreground">{t('home.executionPulse')}</p><h2 className="mt-1 font-semibold">{t('home.taskStatus')}</h2></div></div>
          <div className="mt-6 grid grid-cols-2 gap-3">
            {[
              ['running', t('home.running'), Activity, 'text-cyan-300'],
              ['completed', t('home.completed'), CheckCircle2, 'text-emerald-300'],
              ['completed_empty', t('home.noFactorResult'), AlertTriangle, 'text-amber-300'],
              ['failed', t('home.failed'), XCircle, 'text-red-300'],
            ].map(([status, label, Icon, color]) => {
              const StatusIcon = Icon as typeof Activity;
              return <div key={status as string} className="rounded-lg border border-border bg-secondary/20 p-4"><StatusIcon className={`w-4 h-4 ${color}`} /><p className="mt-4 text-2xl font-bold">{dashboard.tasks.statuses[status as string] || 0}</p><p className="mt-1 text-xs text-muted-foreground">{label as string}</p></div>;
            })}
          </div>
          <Link to="/launchpad" className="mt-5 inline-flex w-full items-center justify-center gap-2 rounded-md border border-border bg-secondary/40 px-3 py-2.5 text-sm font-semibold hover:bg-secondary transition-colors">{t('home.openTracker')} <ArrowRight className="w-4 h-4" /></Link>
        </div>

        <div className="xl:col-span-3 rounded-xl border border-border bg-card overflow-hidden">
          <div className="flex items-center justify-between border-b border-border bg-secondary/20 px-5 py-4"><div className="flex items-center gap-2"><Clock3 className="w-4 h-4 text-primary" /><div><p className="text-xs font-bold uppercase tracking-wider text-muted-foreground">{t('home.latestExecution')}</p><h2 className="mt-1 font-semibold">{t('home.miningActivity')}</h2></div></div><Link to="/launchpad" className="text-xs text-primary hover:underline">{t('home.seeAllTasks')}</Link></div>
          <div className="divide-y divide-border">
            {dashboard.tasks.recent.map((task) => <Link key={task.id} to="/launchpad" className="flex items-center gap-4 px-5 py-4 hover:bg-secondary/30 transition-colors"><span className={`inline-flex min-w-24 justify-center rounded border px-2 py-1 text-[11px] font-semibold ${statusClass(task.status)}`}>{task.status.replace('_', ' ')}</span><div className="min-w-0 flex-1"><div className="flex items-center gap-2"><span className="font-mono text-xs text-foreground">{task.id}</span><span className="rounded bg-secondary px-1.5 py-0.5 text-[10px] text-muted-foreground">{task.miner}</span></div><p className="mt-1 truncate text-xs text-muted-foreground">{task.config} · {task.result_count || 0} factor(s)</p></div><div className="text-right"><p className="text-xs text-foreground">{task.duration}</p><p className="mt-1 text-[11px] text-muted-foreground">{time(task.start_time)}</p></div></Link>)}
            {!loading && dashboard.tasks.recent.length === 0 && <div className="px-5 py-12 text-center text-sm text-muted-foreground">{t('home.noTaskActivity')}</div>}
          </div>
        </div>
      </section>
    </div>
  );
}
