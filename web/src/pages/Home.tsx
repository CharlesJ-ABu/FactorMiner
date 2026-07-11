import { useState, useEffect } from 'react';
import { Activity, Database, Play, Search, CheckCircle, XCircle, Clock } from 'lucide-react';
import { Link } from 'react-router-dom';

export function Home() {
  const [stats, setStats] = useState<any>({
    engine_online: true,
    total_tasks: 0,
    total_factors: 0,
    success_rate: '0%',
    recent_activity: []
  });

  useEffect(() => {
    const fetchStats = () => {
      fetch('http://localhost:8000/api/stats')
        .then(res => res.json())
        .then(data => setStats(data))
        .catch(err => console.error("Failed to fetch stats:", err));
    };

    fetchStats();
    // Refresh stats every 5 seconds
    const interval = setInterval(fetchStats, 5000);
    return () => clearInterval(interval);
  }, []);

  return (
    <div className="flex flex-col h-full gap-8 max-w-6xl mx-auto py-4">
      {/* Header Section */}
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-3xl font-bold tracking-tight text-foreground">Command Center</h1>
          <p className="text-muted-foreground mt-1 text-sm">FactorMiner V4 Global Dashboard</p>
        </div>
        
        <div className="flex items-center gap-3 bg-secondary/30 px-4 py-2 rounded-lg border border-border">
          <div className={`w-3 h-3 rounded-full ${stats.engine_online ? 'bg-green-500 animate-pulse' : 'bg-red-500'}`} />
          <span className="text-sm font-medium">Engine {stats.engine_online ? 'Online' : 'Offline'}</span>
        </div>
      </div>

      {/* Metrics Cards */}
      <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
        <div className="bg-card p-6 rounded-xl border border-border shadow-sm flex flex-col justify-between hover:border-primary/50 transition-colors">
          <div className="flex items-center justify-between mb-4">
            <h3 className="text-muted-foreground font-medium text-sm">Total Tasks Run</h3>
            <Activity className="text-blue-500 w-5 h-5" />
          </div>
          <div className="text-4xl font-bold">{stats.total_tasks}</div>
        </div>

        <div className="bg-card p-6 rounded-xl border border-border shadow-sm flex flex-col justify-between hover:border-primary/50 transition-colors">
          <div className="flex items-center justify-between mb-4">
            <h3 className="text-muted-foreground font-medium text-sm">Mined Factors</h3>
            <Database className="text-green-500 w-5 h-5" />
          </div>
          <div className="text-4xl font-bold">{stats.total_factors}</div>
        </div>

        <div className="bg-card p-6 rounded-xl border border-border shadow-sm flex flex-col justify-between hover:border-primary/50 transition-colors">
          <div className="flex items-center justify-between mb-4">
            <h3 className="text-muted-foreground font-medium text-sm">Success Rate</h3>
            <CheckCircle className="text-purple-500 w-5 h-5" />
          </div>
          <div className="text-4xl font-bold">{stats.success_rate}</div>
        </div>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-3 gap-8 flex-1">
        
        {/* Activity Feed */}
        <div className="lg:col-span-2 flex flex-col bg-card rounded-xl border border-border shadow-sm overflow-hidden">
          <div className="p-5 border-b border-border bg-secondary/20">
            <h2 className="font-semibold flex items-center gap-2">
              <Clock className="w-4 h-4 text-muted-foreground" />
              Recent Mining Activity
            </h2>
          </div>
          <div className="p-5 flex-1 overflow-auto">
            {stats.recent_activity.length === 0 ? (
              <div className="h-full flex items-center justify-center text-muted-foreground text-sm">
                No recent activity.
              </div>
            ) : (
              <div className="space-y-6">
                {stats.recent_activity.map((task: any, index: number) => (
                  <div key={task.id} className="flex gap-4 relative">
                    {/* Timeline Line */}
                    {index !== stats.recent_activity.length - 1 && (
                      <div className="absolute left-4 top-8 bottom-[-24px] w-0.5 bg-border z-0"></div>
                    )}
                    
                    {/* Status Icon */}
                    <div className="relative z-10 flex-shrink-0 mt-1">
                      {task.status === 'completed' && <CheckCircle className="w-8 h-8 text-green-500 bg-background rounded-full" />}
                      {task.status === 'failed' && <XCircle className="w-8 h-8 text-red-500 bg-background rounded-full" />}
                      {task.status === 'running' && <Activity className="w-8 h-8 text-yellow-500 bg-background rounded-full animate-pulse" />}
                    </div>
                    
                    {/* Content */}
                    <div className="flex-1 bg-secondary/10 p-4 rounded-lg border border-border/50">
                      <div className="flex justify-between items-start mb-2">
                        <div>
                          <span className="font-mono text-xs text-muted-foreground">{task.id}</span>
                          <h4 className="font-semibold text-sm mt-1">{task.miner} Mining {task.status === 'running' ? `(${task.progress}%)` : ''}</h4>
                        </div>
                        <span className="text-xs text-muted-foreground">{new Date(task.start_time).toLocaleTimeString()}</span>
                      </div>
                      
                      <p className="text-sm text-muted-foreground">
                        Config: <span className="font-mono text-foreground/80">{task.config}</span>
                      </p>
                      
                      {task.status === 'completed' && task.hash && (
                        <div className="mt-3 text-xs bg-green-950/20 text-green-400 p-2 rounded border border-green-900/30 font-mono">
                          Best Factor Hash: {task.hash}
                        </div>
                      )}
                      
                      {task.status === 'failed' && task.error_msg && (
                        <div className="mt-3 text-xs bg-red-950/20 text-red-400 p-2 rounded border border-red-900/30 overflow-hidden text-ellipsis line-clamp-2">
                          {task.error_msg}
                        </div>
                      )}
                    </div>
                  </div>
                ))}
              </div>
            )}
          </div>
        </div>

        {/* Quick Actions */}
        <div className="flex flex-col gap-4">
          <h2 className="font-semibold mb-2 px-1">Quick Actions</h2>
          
          <Link to="/launchpad" className="group">
            <div className="bg-gradient-to-br from-blue-900/40 to-blue-950/40 p-6 rounded-xl border border-blue-800/50 hover:border-blue-500/50 transition-all hover:shadow-[0_0_20px_rgba(59,130,246,0.15)] flex flex-col items-center justify-center text-center gap-4 h-40">
              <div className="bg-blue-500/20 p-3 rounded-full group-hover:scale-110 transition-transform">
                <Play className="w-8 h-8 text-blue-400 fill-current" />
              </div>
              <div>
                <h3 className="font-bold text-blue-100">Launch New Mining</h3>
                <p className="text-xs text-blue-300/60 mt-1">Configure and start a new job</p>
              </div>
            </div>
          </Link>

          <Link to="/inspector" className="group">
            <div className="bg-gradient-to-br from-purple-900/40 to-purple-950/40 p-6 rounded-xl border border-purple-800/50 hover:border-purple-500/50 transition-all hover:shadow-[0_0_20px_rgba(168,85,247,0.15)] flex flex-col items-center justify-center text-center gap-4 h-40">
              <div className="bg-purple-500/20 p-3 rounded-full group-hover:scale-110 transition-transform">
                <Search className="w-8 h-8 text-purple-400" />
              </div>
              <div>
                <h3 className="font-bold text-purple-100">Inspect Factor DB</h3>
                <p className="text-xs text-purple-300/60 mt-1">Review ASTs and evaluate results</p>
              </div>
            </div>
          </Link>
          
          <Link to="/data" className="group">
            <div className="bg-card p-6 rounded-xl border border-border hover:border-primary/50 transition-all flex flex-col items-center justify-center text-center gap-4 h-40">
              <div className="bg-secondary p-3 rounded-full group-hover:scale-110 transition-transform">
                <Database className="w-8 h-8 text-muted-foreground group-hover:text-primary transition-colors" />
              </div>
              <div>
                <h3 className="font-bold text-foreground">Data Manager</h3>
                <p className="text-xs text-muted-foreground mt-1">Download & align market feeds</p>
              </div>
            </div>
          </Link>
        </div>

      </div>
    </div>
  );
}
