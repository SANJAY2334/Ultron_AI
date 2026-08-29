import React from 'react';
import { Database, Cpu, Activity, Server, Layers, Wifi, WifiOff } from 'lucide-react';
import type { ConnectivityStatus } from '../state/UltronSystemState';
import type { SystemResourceInfo } from '../types';

interface MemoryPanelProps {
  systemTelemetry: SystemResourceInfo;
  connectivity: ConnectivityStatus;
}

export const MemoryPanel: React.FC<MemoryPanelProps> = ({ systemTelemetry, connectivity }) => {
  const isDataAvailable = systemTelemetry.cpu_percent !== 0 || systemTelemetry.memory_used_gb !== 0;
  const memoryPct = systemTelemetry.memory_total_gb > 0 ? Math.round((systemTelemetry.memory_used_gb / systemTelemetry.memory_total_gb) * 100) : 0;
  
  const isConnected = connectivity === 'ONLINE';

  return (
    <div className="glass-panel p-5 flex flex-col h-full">
      <div className="flex items-center justify-between mb-4 border-b border-cyan-500/20 pb-3">
        <div className="flex items-center gap-2">
          <Database className="w-5 h-5 text-purple-400" />
          <h2 className="font-heading text-base font-bold text-cyan-300">
            HYBRID MEMORY MATRIX & SYSTEM HARDWARE
          </h2>
        </div>
        <div className="flex items-center gap-3 text-xs font-code">
          <div className={`flex items-center gap-1 ${isConnected ? 'text-emerald-400' : 'text-red-400'}`}>
            {isConnected ? <Wifi className="w-3.5 h-3.5" /> : <WifiOff className="w-3.5 h-3.5" />}
            <span>{isConnected ? 'CONNECTED' : 'DISCONNECTED'}</span>
          </div>
          <div className="flex items-center gap-1 text-slate-400">
            <Layers className="w-3.5 h-3.5 text-purple-400" />
            <span>PGVECTOR + REDIS</span>
          </div>
        </div>
      </div>

      <div className="grid grid-cols-1 sm:grid-cols-3 gap-3 mb-4 font-code text-xs">
        <div className="bg-slate-950/70 p-3 rounded border border-purple-500/30 relative">
          <div className="flex items-center justify-between text-slate-400 text-[10px] mb-1">
            <span>WORKING MEMORY</span>
            <span className="text-purple-400 font-bold">REDIS</span>
          </div>
          <div className={`font-bold text-sm ${isConnected ? 'text-cyan-300' : 'text-slate-500'}`}>
            {isConnected ? '48 ACTIVE ITEMS' : 'UNAVAILABLE'}
          </div>
          <div className="text-[10px] text-slate-500 mt-1">TTL: Ephemeral Session</div>
        </div>

        <div className="bg-slate-950/70 p-3 rounded border border-purple-500/30 relative">
          <div className="flex items-center justify-between text-slate-400 text-[10px] mb-1">
            <span>EPISODIC MEMORY</span>
            <span className="text-purple-400 font-bold">PGVECTOR</span>
          </div>
          <div className={`font-bold text-sm ${isConnected ? 'text-purple-300' : 'text-slate-500'}`}>
            {isConnected ? '1,240 EMBEDDINGS' : 'UNAVAILABLE'}
          </div>
          <div className="text-[10px] text-slate-500 mt-1">Cosine Similarity Index</div>
        </div>

        <div className="bg-slate-950/70 p-3 rounded border border-purple-500/30 relative">
          <div className="flex items-center justify-between text-slate-400 text-[10px] mb-1">
            <span>KNOWLEDGE GRAPH</span>
            <span className="text-purple-400 font-bold">RELATIONAL</span>
          </div>
          <div className={`font-bold text-sm ${isConnected ? 'text-emerald-300' : 'text-slate-500'}`}>
            {isConnected ? '312 NODES' : 'UNAVAILABLE'}
          </div>
          <div className="text-[10px] text-slate-500 mt-1">Entity-Relation Triples</div>
        </div>
      </div>

      <div className="space-y-3 font-code text-xs bg-slate-950/60 p-4 rounded border border-slate-800">
        <div className="text-slate-300 font-bold mb-2 flex items-center justify-between">
          <span className="flex items-center gap-1.5">
            <Cpu className="w-4 h-4 text-cyan-400" />
            <span>CPU UTILIZATION</span>
          </span>
          <span className="text-cyan-400 font-bold">
            {isDataAvailable ? `${systemTelemetry.cpu_percent}%` : '--'}
          </span>
        </div>
        <div className="w-full h-2 bg-slate-900 rounded-full overflow-hidden border border-slate-800">
          <div
            className="h-full bg-gradient-to-r from-cyan-500 to-purple-500 transition-all duration-500"
            style={{ width: isDataAvailable ? `${systemTelemetry.cpu_percent}%` : '0%' }}
          />
        </div>

        <div className="text-slate-300 font-bold mb-2 pt-2 flex items-center justify-between">
          <span className="flex items-center gap-1.5">
            <Activity className="w-4 h-4 text-purple-400" />
            <span>RAM MEMORY ALLOCATION</span>
          </span>
          <span className="text-purple-400 font-bold">
            {isDataAvailable ? `${systemTelemetry.memory_used_gb} GB (${memoryPct}%)` : '--'}
          </span>
        </div>
        <div className="w-full h-2 bg-slate-900 rounded-full overflow-hidden border border-slate-800">
          <div
            className="h-full bg-gradient-to-r from-purple-500 to-emerald-500 transition-all duration-500"
            style={{ width: isDataAvailable ? `${memoryPct}%` : '0%' }}
          />
        </div>

        <div className="pt-2 flex items-center justify-between text-[11px] text-slate-400 border-t border-slate-800/80">
          <span className="flex items-center gap-1">
            <Server className="w-3.5 h-3.5 text-slate-500" />
            <span>ACTIVE PROCESSES: {isDataAvailable ? systemTelemetry.active_processes : '--'}</span>
          </span>
          <span>UPTIME: {isDataAvailable ? `${Math.floor(systemTelemetry.uptime_seconds / 3600)}h ${Math.floor((systemTelemetry.uptime_seconds % 3600) / 60)}m` : '--'}</span>
        </div>
      </div>
    </div>
  );
};
