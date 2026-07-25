import { useState, useEffect, useCallback } from 'react';
import { Download, CheckCircle2, Info, Loader2 } from 'lucide-react';
import { useWebSocket } from '../hooks/useWebSocket';
import Slider from 'rc-slider';
import 'rc-slider/assets/index.css';

export function DataDownload() {
  const [exchange, setExchange] = useState('binance');
  
  // Meta state
  const [meta, setMeta] = useState<any>({
    symbols: ["BTC/USDT", "ETH/USDT"],
    timeframes: ["1m", "1d"],
    trade_types: ["spot", "futures"],
    min_date: "2017-01-01"
  });
  const [loadingMeta, setLoadingMeta] = useState(false);

  // Form state (Multiple selection for symbols and timeframes)
  const [symbols, setSymbols] = useState<string[]>(['BTC/USDT']);
  const [customSymbol, setCustomSymbol] = useState('');
  
  const [timeframes, setTimeframes] = useState<string[]>(['1d']);
  const [tradeType, setTradeType] = useState<string>('futures');
  
  // Date Slider state (using timestamps for the slider)
  const [dateRange, setDateRange] = useState<number[]>([
    new Date('2023-01-01').getTime(),
    new Date().getTime()
  ]);
  
  const [downloadMode, setDownloadMode] = useState('merge');
  
  const [isDownloading, setIsDownloading] = useState(false);
  const [progress, setProgress] = useState(0);
  const [downloadMsg, setDownloadMsg] = useState('');
  const [logs, setLogs] = useState<string[]>([]);
  
  const [coverageResults, setCoverageResults] = useState<any[]>([]);

  const { lastMessage } = useWebSocket('ws://localhost:8000/ws/monitor');

  // Fetch Meta
  useEffect(() => {
    setLoadingMeta(true);
    fetch(`http://localhost:8000/api/exchange_meta?exchange=${exchange}&trade_type=${tradeType}`)
      .then(res => res.json())
      .then(data => {
        setMeta(data);
        // Reset selections if they are not in the new meta
        setSymbols(prev => prev.filter(p => data.symbols.includes(p)));
        setTimeframes(prev => prev.filter(p => data.timeframes.includes(p)));
        // Note: tradeType is now single-select, so we just keep it
      })
      .finally(() => setLoadingMeta(false));
  }, [exchange, tradeType]);

  // Fetch Coverage
  const fetchCoverage = useCallback(() => {
    if (symbols.length === 0 || timeframes.length === 0) {
      setCoverageResults([]);
      return;
    }
    fetch('http://localhost:8000/api/batch_data_coverage', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        exchange,
        symbols,
        timeframes,
        trade_types: [tradeType]
      })
    })
      .then(res => res.json())
      .then(data => setCoverageResults(data.results || []))
      .catch(err => console.error("Failed to fetch coverage", err));
  }, [exchange, symbols, timeframes, tradeType]);

  // Debounce coverage fetching to avoid spamming the backend while clicking
  useEffect(() => {
    const timer = setTimeout(() => {
      fetchCoverage();
    }, 500);
    return () => clearTimeout(timer);
  }, [fetchCoverage]);

  useEffect(() => {
    if (lastMessage?.type === 'download_progress') {
      setProgress(lastMessage.progress);
      setDownloadMsg(lastMessage.message);
      setLogs(prev => [...prev.slice(-99), lastMessage.message]);
      if (lastMessage.progress >= 100) {
        setIsDownloading(false);
        fetchCoverage();
      }
    }
  }, [lastMessage, fetchCoverage]);

  const handleDownload = () => {
    if (symbols.length === 0 || timeframes.length === 0 || !tradeType) {
      alert("Please select at least one Symbol, Timeframe, and Trade Type.");
      return;
    }

    setIsDownloading(true);
    setProgress(0);
    setDownloadMsg('Starting download...');
    setLogs(['> Initializing batch download job...', '> Contacting server...']);
    
    // Format dates back to YYYY-MM-DD
    const startDate = new Date(dateRange[0]).toISOString().split('T')[0];
    const endDate = new Date(dateRange[1]).toISOString().split('T')[0];

    fetch('http://localhost:8000/api/download_data', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        exchange,
        symbols,
        timeframes,
        start_date: startDate,
        end_date: endDate,
        trade_types: [tradeType],
        download_mode: downloadMode
      })
    }).catch(err => {
      console.error(err);
      setIsDownloading(false);
      setDownloadMsg('Failed to start download');
    });
  };

  const toggleSelection = (setter: any, value: string) => {
    setter((prev: string[]) => 
      prev.includes(value) ? prev.filter(v => v !== value) : [...prev, value]
    );
  };

  const handleAddCustomSymbol = () => {
    const sym = customSymbol.trim().toUpperCase();
    if (!sym) return;
    
    // Add to meta if not exists
    if (!meta.symbols.includes(sym)) {
      setMeta((prev: any) => ({
        ...prev,
        symbols: [sym, ...prev.symbols]
      }));
    }
    
    // Select it
    if (!symbols.includes(sym)) {
      setSymbols(prev => [sym, ...prev]);
    }
    
    setCustomSymbol('');
  };

  const minDateTs = new Date(meta.min_date || '2017-01-01').getTime();
  const maxDateTs = new Date().getTime();

  return (
    <div className="flex flex-col h-full gap-6 w-full pb-10">
      <div className="border border-border bg-card rounded-xl p-8 flex flex-col">

        <div className="flex flex-col gap-8 mb-8">
          {/* Top Row: Exchange, Trade Types, Timeframes */}
          <div className="flex flex-col md:flex-row gap-16 items-start w-full">
            {/* Exchange Selection */}
            <div className="w-64 flex-shrink-0 space-y-2">
              <label className="text-sm font-medium text-muted-foreground flex items-center gap-2">
                Exchange
                {loadingMeta && <Loader2 className="w-4 h-4 animate-spin text-primary" />}
              </label>
              <select 
                value={exchange}
                onChange={(e) => setExchange(e.target.value)}
                className="w-full bg-background border border-border rounded-md px-3 py-2 text-foreground focus:outline-none focus:border-primary transition-colors"
              >
                <option value="binance">Binance</option>
                <option value="okx">OKX</option>
              </select>
            </div>
            {/* Trade Types */}
            <div className="flex-shrink-0 space-y-3">
              <label className="text-sm font-medium text-muted-foreground flex justify-between">
                <span>Trade Type (Single Select)</span>
              </label>
              <div className="flex flex-wrap gap-2">
                {meta.trade_types?.map((tt: string) => (
                  <button
                    key={tt}
                    onClick={() => setTradeType(tt)}
                    className={`px-4 py-2 text-sm rounded-md border transition-colors ${
                      tradeType === tt
                        ? 'bg-primary text-primary-foreground border-primary' 
                        : 'bg-secondary text-secondary-foreground border-border hover:border-primary/50'
                    }`}
                  >
                    {tt.charAt(0).toUpperCase() + tt.slice(1)}
                  </button>
                ))}
              </div>
            </div>

            {/* Timeframes */}
            <div className="flex-1 space-y-3">
              <label className="text-sm font-medium text-muted-foreground flex justify-between">
                <span>Timeframes (Multi-Select)</span>
              </label>
              <div className="flex flex-wrap gap-2">
                {meta.timeframes?.map((tf: string) => (
                  <button
                    key={tf}
                    onClick={() => toggleSelection(setTimeframes, tf)}
                    className={`px-3 py-1.5 text-xs font-mono rounded-md border transition-colors ${
                      timeframes.includes(tf) 
                        ? 'bg-primary text-primary-foreground border-primary' 
                        : 'bg-secondary text-secondary-foreground border-border hover:border-primary/50'
                    }`}
                  >
                    {tf}
                  </button>
                ))}
              </div>
            </div>
          </div>

          {/* Bottom Row: Symbols */}
          <div className="w-full">
            <div className="space-y-3">
              <label className="text-sm font-medium text-muted-foreground flex justify-between items-center">
                <span>Symbols (Multi-Select)</span>
                <span className="text-xs text-primary">{symbols.length} selected</span>
              </label>
              
              <div className="flex gap-2 mb-2">
                <input 
                  type="text" 
                  value={customSymbol}
                  onChange={(e) => setCustomSymbol(e.target.value)}
                  onKeyDown={(e) => e.key === 'Enter' && handleAddCustomSymbol()}
                  placeholder="Add custom (e.g. PEPE/USDT)"
                  className="flex-1 bg-background border border-border rounded-md px-3 py-1.5 text-sm text-foreground focus:outline-none focus:border-primary transition-colors"
                />
                <button 
                  onClick={handleAddCustomSymbol}
                  className="px-4 py-1.5 bg-secondary text-secondary-foreground text-sm font-medium rounded-md hover:bg-secondary/80 transition-colors"
                >
                  Add
                </button>
              </div>
              
              <div className="h-32 overflow-y-auto border border-border rounded-md p-2 bg-background flex flex-wrap gap-2 content-start">
                {meta.symbols?.map((sym: string) => (
                  <button
                    key={sym}
                    onClick={() => toggleSelection(setSymbols, sym)}
                    className={`px-3 py-1 text-xs rounded-full border transition-colors ${
                      symbols.includes(sym) 
                        ? 'bg-primary text-primary-foreground border-primary' 
                        : 'bg-secondary text-secondary-foreground border-border hover:border-primary/50'
                    }`}
                  >
                    {sym}
                  </button>
                ))}
              </div>
            </div>
          </div>

          {/* Date Range Slider */}
          <div className="space-y-6 pt-4 border-t border-border">
            <div className="flex justify-between items-end">
              <label className="text-sm font-medium text-muted-foreground">Historical Date Range</label>
              <div className="text-sm font-mono text-foreground bg-secondary px-3 py-1 rounded-md border border-border">
                {new Date(dateRange[0]).toISOString().split('T')[0]} 
                <span className="text-muted-foreground mx-2">to</span> 
                {new Date(dateRange[1]).toISOString().split('T')[0]}
              </div>
            </div>
            
            <div className="px-2">
              <Slider 
                range
                min={minDateTs}
                max={maxDateTs}
                step={86400000} // 1 day in ms
                value={dateRange}
                onChange={(val) => setDateRange(val as number[])}
                styles={{
                  track: { backgroundColor: 'hsl(var(--primary))' },
                  handle: { borderColor: 'hsl(var(--primary))', backgroundColor: 'hsl(var(--background))' },
                  rail: { backgroundColor: 'hsl(var(--border))' }
                }}
              />
            </div>
          </div>

          {/* Download Strategy */}
          <div className="space-y-2 pt-4 border-t border-border">
            <label className="text-sm font-medium text-muted-foreground">Download Strategy</label>
            <select 
              value={downloadMode}
              onChange={(e) => setDownloadMode(e.target.value)}
              className="w-full max-w-md bg-background border border-border rounded-md px-3 py-2 text-foreground focus:outline-none focus:border-primary transition-colors"
            >
              <option value="merge">Smart Merge & Deduplicate (Recommended)</option>
              <option value="fill_gap">Fill Gap & Merge (Auto-extend)</option>
              <option value="overwrite">Force Overwrite Local Data</option>
            </select>
          </div>
        </div>

        {/* Dynamic Display Area: Coverage OR Console */}
        {isDownloading || progress === 100 ? (
          <div className="mb-8 p-4 bg-[#0a0a0a] rounded-lg border border-border flex flex-col max-h-60 overflow-hidden shadow-inner">
            <div className="flex items-center gap-2 text-primary font-bold text-sm mb-3 border-b border-border/50 pb-2">
              {isDownloading ? <Loader2 className="w-4 h-4 animate-spin" /> : <CheckCircle2 className="w-4 h-4" />}
              Execution Console
            </div>
            <div className="overflow-y-auto font-mono text-xs text-muted-foreground flex flex-col gap-1 pr-2">
              {logs.map((log, idx) => (
                <div key={idx} className="hover:text-foreground transition-colors break-words">
                  {log}
                </div>
              ))}
              {logs.length === 0 && <div className="text-muted-foreground/50 italic">Waiting for log stream...</div>}
            </div>
          </div>
        ) : (
          <div className="mb-8 p-4 border border-dashed border-primary/50 bg-primary/5 rounded-lg flex flex-col gap-4 max-h-60 overflow-y-auto">
            <div className="flex items-center justify-between sticky top-0 bg-primary/5 pb-2">
              <div className="flex items-center gap-2 text-primary font-bold text-sm">
                <Info className="w-4 h-4" />
                Batch Local Coverage ({coverageResults.length} targets)
              </div>
            </div>
            
            <div className="grid grid-cols-1 md:grid-cols-2 gap-3">
              {coverageResults.map((res, idx) => (
                <div key={idx} className="bg-background/50 border border-primary/20 rounded-md p-3 text-sm flex flex-col gap-1">
                  <div className="flex justify-between font-bold text-foreground">
                    <span>{res.symbol} <span className="text-xs text-muted-foreground font-normal ml-1">({res.trade_type})</span></span>
                    <span className="font-mono bg-secondary px-2 rounded text-xs leading-5">{res.timeframe}</span>
                  </div>
                  {res.coverage?.exists ? (
                    <div className="text-xs text-muted-foreground mt-1 space-y-1">
                      <div className="flex justify-between">
                        <span>Range:</span>
                        <span className="font-mono text-primary/80">{res.coverage.start_date.split(' ')[0]} → {res.coverage.end_date.split(' ')[0]}</span>
                      </div>
                      <div className="flex justify-between">
                        <span>Records:</span>
                        <span className="font-mono">{res.coverage.total_records.toLocaleString()}</span>
                      </div>
                    </div>
                  ) : (
                    <div className="text-xs text-muted-foreground mt-1 flex h-full items-center">
                      No local data found. Will fetch fresh.
                    </div>
                  )}
                </div>
              ))}
              
              {coverageResults.length === 0 && (
                <div className="text-sm text-muted-foreground italic col-span-full">
                  Please select targets to view local coverage...
                </div>
              )}
            </div>
          </div>
        )}

        <div className="mt-auto">
          {isDownloading ? (
            <div className="space-y-2">
              <div className="flex justify-between text-sm text-muted-foreground">
                <span className="truncate pr-4">{downloadMsg}</span>
                <span>{progress}%</span>
              </div>
              <div className="h-2 w-full bg-secondary rounded-full overflow-hidden">
                <div 
                  className="h-full bg-primary transition-all duration-200"
                  style={{ width: `${progress}%` }}
                />
              </div>
            </div>
          ) : progress === 100 ? (
            <div className="flex flex-col gap-3">
              <div className="flex items-center gap-2 p-4 bg-green-500/10 text-green-500 border border-green-500/20 rounded-lg">
                <CheckCircle2 className="w-5 h-5" />
                <span>Successfully completed batch download!</span>
              </div>
              <button 
                onClick={() => setProgress(0)}
                className="w-full text-sm text-muted-foreground hover:text-foreground underline transition-colors"
              >
                Start another batch
              </button>
            </div>
          ) : (
            <button 
              onClick={handleDownload}
              className="w-full flex items-center justify-center gap-2 bg-primary text-primary-foreground py-3 rounded-md font-bold hover:opacity-90 transition-opacity disabled:opacity-50 disabled:cursor-not-allowed"
              disabled={symbols.length === 0 || timeframes.length === 0 || !tradeType}
            >
              <Download className="w-5 h-5" /> Execute Batch Download
            </button>
          )}
        </div>
      </div>
    </div>
  );
}
