import React, { Component, useState } from 'react';
import type { ErrorInfo, ReactNode } from 'react';
import { Maximize2, Minimize2, ShieldCheck, Activity, Cpu } from 'lucide-react';
import type { AvatarState, AvatarViewMode } from './types';
import { RoboticFace } from './RoboticFace';
import { FallbackFace } from './FallbackFace';
import { getAvatarColor } from './stateMapper';

interface ErrorBoundaryProps {
  children: ReactNode;
  fallback: ReactNode;
}

interface ErrorBoundaryState {
  hasError: boolean;
}

class AvatarErrorBoundary extends Component<ErrorBoundaryProps, ErrorBoundaryState> {
  public state: ErrorBoundaryState = { hasError: false };

  public static getDerivedStateFromError(_: Error): ErrorBoundaryState {
    return { hasError: true };
  }

  public componentDidCatch(error: Error, errorInfo: ErrorInfo) {
    console.error('Avatar rendering error caught:', error, errorInfo);
  }

  public render() {
    if (this.state.hasError) {
      return this.props.fallback;
    }
    return this.props.children;
  }
}

interface AvatarContainerProps {
  state: AvatarState;
  viewMode?: AvatarViewMode;
  onViewModeChange?: (mode: AvatarViewMode) => void;
}

export const AvatarContainer: React.FC<AvatarContainerProps> = ({
  state,
  viewMode = 'EMBEDDED',
  onViewModeChange,
}) => {
  const [currentMode, setCurrentMode] = useState<AvatarViewMode>(viewMode);
  const color = getAvatarColor(state);

  const handleModeChange = (mode: AvatarViewMode) => {
    setCurrentMode(mode);
    if (onViewModeChange) onViewModeChange(mode);
  };

  if (currentMode === 'COMPACT') {
    return (
      <div className="flex items-center gap-2 bg-slate-950/90 px-3 py-1.5 rounded-full border border-cyan-500/40 shadow-lg">
        <RoboticFace state={state} compact={true} />
        <div className="flex flex-col">
          <span className="text-[10px] font-code font-bold text-cyan-300">ULTRON AVATAR</span>
          <span className="text-[9px] font-code font-bold" style={{ color: color.primary }}>
            {state}
          </span>
        </div>
        <button
          onClick={() => handleModeChange('EMBEDDED')}
          className="ml-2 text-slate-400 hover:text-cyan-400 text-xs font-code p-1"
          title="Expand View"
        >
          <Maximize2 className="w-3.5 h-3.5" />
        </button>
      </div>
    );
  }

  if (currentMode === 'FULLSCREEN') {
    return (
      <div className="fixed inset-0 z-50 bg-slate-950/95 backdrop-blur-xl flex flex-col items-center justify-between p-8 border-4 border-cyan-500/40">
        {/* Fullscreen Header */}
        <div className="w-full max-w-5xl flex items-center justify-between font-code border-b border-cyan-500/30 pb-4">
          <div className="flex items-center gap-3">
            <Cpu className="w-6 h-6 text-cyan-400 animate-pulse" />
            <div>
              <h2 className="font-heading text-lg font-bold text-cyan-300 tracking-widest">
                ULTRON ROBOTIC AVATAR — FULLSCREEN MODE
              </h2>
              <span className="text-xs text-slate-400">ISOLATED FRONTEND VISUALIZATION LAYER</span>
            </div>
          </div>

          <div className="flex items-center gap-4">
            <span className={`px-3 py-1 rounded text-xs font-bold border ${color.badge}`}>
              SYSTEM STATE: {state}
            </span>
            <button
              onClick={() => handleModeChange('EMBEDDED')}
              className="flex items-center gap-1.5 px-3 py-1.5 bg-slate-900 hover:bg-slate-800 border border-slate-700 text-slate-200 text-xs font-code rounded transition-all"
            >
              <Minimize2 className="w-4 h-4" />
              EXIT FULLSCREEN
            </button>
          </div>
        </div>

        {/* Center Robotic Face */}
        <div className="my-auto w-full max-w-[500px]">
          <AvatarErrorBoundary fallback={<FallbackFace state={state} />}>
            <RoboticFace state={state} />
          </AvatarErrorBoundary>
        </div>

        {/* Fullscreen Footer Privacy Invariant Badge */}
        <div className="w-full max-w-5xl flex items-center justify-between text-xs font-code text-slate-400 border-t border-slate-900 pt-4">
          <div className="flex items-center gap-2">
            <ShieldCheck className="w-4 h-4 text-emerald-400" />
            <span>ZERO SENSITIVE LEAKAGE (NO CREDENTIALS / RAW FRAMES / CHAIN-OF-THOUGHT)</span>
          </div>
          <div className="flex items-center gap-2">
            <Activity className="w-4 h-4 text-cyan-400" />
            <span>GPU RENDERER: SVG + CSS ACCELERATED</span>
          </div>
        </div>
      </div>
    );
  }

  // Default: EMBEDDED DASHBOARD MODE
  return (
    <div className="glass-panel p-5 flex flex-col items-center justify-between h-full relative overflow-hidden">
      {/* Header Controls */}
      <div className="w-full flex items-center justify-between border-b border-cyan-500/20 pb-3 mb-2">
        <div className="flex items-center gap-2">
          <Cpu className="w-5 h-5 text-cyan-400" />
          <h2 className="font-heading text-base font-bold text-cyan-300">
            ROBOTIC AVATAR RENDERER
          </h2>
        </div>
        <div className="flex items-center gap-2">
          <span className={`px-2.5 py-0.5 rounded text-[11px] font-code font-bold border ${color.badge}`}>
            {state}
          </span>
          <button
            onClick={() => handleModeChange('FULLSCREEN')}
            className="p-1 rounded bg-slate-900 text-slate-400 hover:text-cyan-300 border border-slate-800 hover:border-cyan-500/40 text-xs font-code transition-all"
            title="Fullscreen Mode"
          >
            <Maximize2 className="w-3.5 h-3.5" />
          </button>
        </div>
      </div>

      {/* Main Robotic Face Viewport */}
      <div className="w-full my-auto flex items-center justify-center py-2">
        <AvatarErrorBoundary fallback={<FallbackFace state={state} />}>
          <RoboticFace state={state} />
        </AvatarErrorBoundary>
      </div>

      {/* Privacy Guard & State Indicator Bar */}
      <div className="w-full pt-3 border-t border-slate-900 flex items-center justify-between text-[11px] font-code text-slate-400">
        <div className="flex items-center gap-1.5 text-emerald-400">
          <ShieldCheck className="w-3.5 h-3.5" />
          <span>VISUALIZATION ONLY</span>
        </div>
        <div className="text-slate-500 text-[10px]">
          FROZEN ZERO-TRUST BOUNDARY
        </div>
      </div>
    </div>
  );
};
