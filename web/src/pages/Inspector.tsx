import { useEffect, useRef } from 'react';
import ReactECharts from 'echarts-for-react';

import { Box, Code } from 'lucide-react';

export function Inspector() {
  const treeData = {
    name: 'cross_sectional_rank',
    children: [
      {
        name: 'ts_mean(10)',
        children: [{ name: 'close_price' }]
      },
      {
        name: 'ts_std(20)',
        children: [{ name: 'volume' }]
      }
    ]
  };

  const treeOption = {
    backgroundColor: 'transparent',
    tooltip: { trigger: 'item', triggerOn: 'mousemove' },
    series: [
      {
        type: 'tree',
        data: [treeData],
        top: '10%', left: '20%', bottom: '10%', right: '20%',
        symbolSize: 10,
        itemStyle: { color: '#3b82f6', borderColor: '#60a5fa' },
        lineStyle: { color: '#4b5563', width: 2, curveness: 0.5 },
        label: {
          position: 'top',
          verticalAlign: 'middle',
          align: 'center',
          fontSize: 12,
          color: '#e5e7eb',
          backgroundColor: '#1f2937',
          padding: [4, 8],
          borderRadius: 4,
          borderColor: '#3b82f6',
          borderWidth: 1
        },
        leaves: {
          label: {
            position: 'bottom',
            verticalAlign: 'middle',
            align: 'center'
          }
        },
        expandAndCollapse: true,
        animationDuration: 550,
        animationDurationUpdate: 750
      }
    ]
  };

  const netValueOption = {
    backgroundColor: 'transparent',
    grid: { top: 40, right: 20, bottom: 40, left: 50 },
    tooltip: { trigger: 'axis', axisPointer: { type: 'cross' } },
    xAxis: { type: 'category', data: Array.from({length: 100}, (_, i) => `D${i+1}`), axisLabel: { color: '#888' } },
    yAxis: { type: 'value', splitLine: { lineStyle: { color: 'rgba(255,255,255,0.1)' } }, axisLabel: { color: '#888' }, min: 'dataMin' },
    series: [
      {
        name: 'Net Value',
        type: 'line',
        data: Array.from({length: 100}, () => Math.random() * 0.1 + 1).reduce((acc: number[], val) => {
          acc.push((acc[acc.length - 1] || 1) * val);
          return acc;
        }, []),
        smooth: true,
        itemStyle: { color: '#00ff9d' },
        areaStyle: {
          color: {
            type: 'linear', x: 0, y: 0, x2: 0, y2: 1,
            colorStops: [{ offset: 0, color: 'rgba(0, 255, 157, 0.3)' }, { offset: 1, color: 'rgba(0, 255, 157, 0)' }]
          }
        }
      }
    ]
  };

  return (
    <div className="flex flex-col h-full gap-6">
      {/* Header Badges */}
      <div className="flex items-center gap-4">
        <h2 className="text-xl font-bold">fac_gp_alpha_001</h2>
        <span className="px-3 py-1 bg-green-500/20 text-green-400 border border-green-500/50 rounded-full text-xs font-bold">🔍 APPROVED</span>
        <span className="px-3 py-1 bg-blue-500/20 text-blue-400 border border-blue-500/50 rounded-full text-xs font-bold">GP Paradigm</span>
      </div>

      <div className="grid grid-cols-2 gap-6 h-80">
        {/* AST Tree */}
        <div className="border border-border bg-card rounded-xl p-4 flex flex-col">
          <div className="flex items-center gap-2 mb-4 text-sm font-bold text-foreground">
            <Box className="w-4 h-4 text-primary" /> Logic Reference (AST)
          </div>
          <div className="flex-1 w-full min-h-0">
            <ReactECharts option={treeOption} style={{ height: '100%', width: '100%' }} />
          </div>
        </div>

        {/* Code / Logic String */}
        <div className="border border-border bg-card rounded-xl p-4 flex flex-col">
          <div className="flex items-center gap-2 mb-4 text-sm font-bold text-foreground">
            <Code className="w-4 h-4 text-primary" /> Expression
          </div>
          <div className="flex-1 bg-black rounded-lg p-4 font-mono text-sm text-green-400 overflow-auto border border-border">
            cross_sectional_rank(<br/>
            &nbsp;&nbsp;ts_mean(close_price, 10),<br/>
            &nbsp;&nbsp;ts_std(volume, 20)<br/>
            )
          </div>
        </div>
      </div>

      {/* Tearsheet */}
      <div className="flex-1 border border-border bg-card rounded-xl p-4 flex flex-col">
        <h2 className="text-sm font-bold text-foreground mb-4">📊 Tearsheet Performance</h2>
        <div className="flex-1 w-full min-h-0">
          <ReactECharts option={netValueOption} style={{ height: '100%', width: '100%' }} />
        </div>
      </div>
    </div>
  );
}
