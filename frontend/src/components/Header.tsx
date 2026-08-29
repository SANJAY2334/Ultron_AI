import React from 'react';
import { ShieldCheck, ShieldAlert, Cpu, Activity, Radio, Zap, Minus, Square, X } from 'lucide-react';
import type { SystemResourceInfo } from '../types';
import type { ConnectivityStatus } from '../state/UltronSystemState';
import type { AvatarState } from '../avatar/types';

interface HeaderProps {
  connectivity: ConnectivityStatus;
  systemTelemetry: SystemResourceInfo;
  avatarState: AvatarState;
  backendVersion: string | null;
  backendKernelState: string | null;
}

export const Header: React.FC<HeaderProps> = ({
  connectivity,
  systemTelemetry,
  avatarState,
  backendVersion,
  backendKernelState,
}) => {
  const isOnline = connectivity === 'ONLINE';
  const isNative = typeof window !== 'undefined' && Boolean(window.ultronNative);

  const handleMinimize = () => {
    window.ultronNative?.minimizeWindow();
  };

  const handleMaximize = () => {
    window.ultronNative?.maximizeWindow();
  };

  const handleCloseToTray = () => {
    window.ultronNative?.closeToTray();
  };

  return (
    <header className="glass-panel px-6 py-4 flex flex-col md:flex-row items-center justify-between gap-4 mb-6 border-b border-cyan-500/30">
      {/* Brand & Logo */}
      <div className="flex items-center gap-3">
        <div className="relative flex items-center justify-center w-10 h-10 rounded-full bg-cyan-950/80 border border-cyan-400/60 shadow-[0_0_15px_rgba(0,240,255,0.4)]">
          <Zap className="w-5 h-5 text-cyan-400 animate-pulse" />
          <div className="absolute inset-0 rounded-full border border-cyan-400/40 animate-ping opacity-25" />
        </div>
        <div>
          <div className="flex items-center gap-2">
            <h1 className="font-heading text-xl font-bold tracking-widest neon-text-cyan">
              ULTRON
            </h1>
            <span className="text-[10px] font-code px-2 py-0.5 rounded bg-cyan-950/80 text-cyan-300 border border-cyan-500/40">
              v4G.4 NATIVE DESKTOP
            </span>
          </div>
          <p className="text-xs text-slate-400 font-code flex items-center gap-2 mt-0.5">
            <span>ZERO-TRUST DOMAIN</span>
            <span className="text-cyan-500">•</span>
            <span>HYBRID MEMORY MATRIX</span>
            {backendVersion && (
              <>
                <span className="text-cyan-500">•</span>
                <span>{backendVersion}</span>
              </>
            )}
            {backendKernelState && (
              <>
                <span className="text-cyan-500">•</span>
                <span>{backendKernelState}</span>
              </>
            )}
          </p>
        </div>
      </div>

      {/* Real-time System Quick Telemetry */}
      <div className="hidden lg:flex items-center gap-6 text-xs font-code bg-slate-950/60 px-4 py-2 rounded-md border border-slate-800">
        <div className="flex items-center gap-2">
          <Cpu className="w-4 h-4 text-cyan-400" />
          <span className="text-slate-400">CPU:</span>
          <span className="text-cyan-300 font-semibold">{systemTelemetry.cpu_percent === 0 ? '--' : systemTelemetry.cpu_percent}%</span>
        </div>
        <div className="w-px h-4 bg-slate-800" />
        <div className="flex items-center gap-2">
          <Activity className="w-4 h-4 text-purple-400" />
          <span className="text-slate-400">RAM:</span>
          <span className="text-purple-300 font-semibold">
            {systemTelemetry.memory_used_gb === 0 ? '--' : systemTelemetry.memory_used_gb} / {systemTelemetry.memory_total_gb === 0 ? '--' : systemTelemetry.memory_total_gb} GB
          </span>
        </div>
        <div className="w-px h-4 bg-slate-800" />
        <div className="flex items-center gap-2">
          <Radio className="w-4 h-4 text-emerald-400" />
          <span className="text-slate-400">PROCS:</span>
          <span className="text-emerald-300 font-semibold">{systemTelemetry.active_processes === 0 ? '--' : systemTelemetry.active_processes}</span>
        </div>
      </div>

      {/* Security Interlock, Connectivity & Native Controls */}
      <div className="flex items-center gap-3 sm:gap-4 flex-wrap">
        {/* Avatar State Badge */}
        <div className="flex items-center gap-2 px-3 py-1.5 rounded-md bg-purple-950/60 border border-purple-500/40 text-purple-300 text-xs font-code uppercase">
          <span className="hidden sm:inline">AVATAR:</span>
          <span className="font-bold">{avatarState}</span>
        </div>

        {/* Policy Engine Guard Badge */}
        <div className={`flex items-center gap-2 px-3 py-1.5 rounded-md border text-xs font-code ${isOnline ? 'bg-emerald-950/60 border-emerald-500/40 text-emerald-300' : 'bg-red-950/60 border-red-500/40 text-red-300'}`}>
          {isOnline ? <ShieldCheck className="w-4 h-4 text-emerald-400" /> : <ShieldAlert className="w-4 h-4 text-red-400" />}
          <span className="hidden sm:inline">POLICY ENGINE:</span>
          <span className="font-bold">{isOnline ? 'ACTIVE' : 'OFFLINE'}</span>
        </div>

        {/* Connectivity Status Indicator */}
        <div className="flex items-center gap-1.5 text-xs font-code bg-slate-950 p-1.5 rounded-lg border border-slate-800">
          <span
            className={`w-2.5 h-2.5 rounded-full ${
              connectivity === 'ONLINE' ? 'bg-emerald-400 shadow-[0_0_8px_#00ff66]' :
              connectivity === 'CONNECTING' ? 'bg-amber-400 animate-pulse' :
              connectivity === 'RECONNECTING' ? 'bg-amber-500 animate-bounce' :
              'bg-red-500'
            }`}
          />
          <span className="text-slate-400 px-1 font-medium hidden xl:inline">
            {connectivity === 'ONLINE' ? 'ONLINE' :
             connectivity === 'CONNECTING' ? 'CONNECTING...' :
             connectivity === 'RECONNECTING' ? 'RECONNECTING...' : 'OFFLINE'}
          </span>
        </div>

        {/* Native Electron Window Controls (Only rendered in Desktop context) */}
        {isNative && (
          <div
            className="flex items-center gap-1 border-l border-slate-800 pl-3 ml-1"
            style={{ WebkitAppRegion: 'no-drag' } as React.CSSProperties}
          >
            <button
              onClick={handleMinimize}
              title="Minimize Window"
              aria-label="Minimize Window"
              className="p-1.5 rounded bg-slate-900/80 hover:bg-slate-800 text-slate-400 hover:text-cyan-300 border border-slate-700/60 transition-colors"
            >
              <Minus className="w-3.5 h-3.5" />
            </button>
            <button
              onClick={handleMaximize}
              title="Maximize or Restore Window"
              aria-label="Maximize or Restore Window"
              className="p-1.5 rounded bg-slate-900/80 hover:bg-slate-800 text-slate-400 hover:text-cyan-300 border border-slate-700/60 transition-colors"
            >
              <Square className="w-3 h-3" />
            </button>
            <button
              onClick={handleCloseToTray}
              title="Close to System Tray"
              aria-label="Close to System Tray"
              className="p-1.5 rounded bg-slate-900/80 hover:bg-red-950/80 text-slate-400 hover:text-red-300 border border-slate-700/60 hover:border-red-500/50 transition-colors"
            >
              <X className="w-3.5 h-3.5" />
            </button>
          </div>
        )}
      </div>
    </header>
  );
};
