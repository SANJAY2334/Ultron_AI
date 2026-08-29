/* ULTRON Native Desktop HUD Overlay Component (Phase 4G.4: Stage 5)
 *
 * Lightweight, compact desktop HUD widget presenting avatar state, live VAD activity,
 * and quick telemetry without opening the full Command Center.
 *
 * HARD SECURITY INVARIANTS:
 * - Avatar != Authority
 * - HUD != ToolExecutor
 * - HUD != PolicyEngine
 * - Presentation ONLY: Zero privileged execution or policy evaluation.
 */

import React from 'react';
import { Cpu, Activity, Radio, Maximize2, X, Zap } from 'lucide-react';
import { AvatarContainer } from '../avatar/AvatarContainer';
import type { UltronSystemState } from '../state/UltronSystemState';
import { getAvatarColor } from '../avatar/stateMapper';

interface HUDOverlayProps {
  state: UltronSystemState;
}

export const HUDOverlay: React.FC<HUDOverlayProps> = ({ state }) => {
  const isOnline = state.connectivity === 'ONLINE';
  const colorTheme = getAvatarColor(state.avatarState);

  const handleOpenCommandCenter = () => {
    if (window.ultronNative?.maximizeWindow) {
      window.ultronNative.maximizeWindow();
    }
  };

  const handleCloseHUD = () => {
    if (window.ultronNative?.toggleHUD) {
      window.ultronNative.toggleHUD();
    }
  };

  return (
    <div className="w-full h-screen bg-slate-950/85 backdrop-blur-md text-slate-100 p-4 border border-cyan-500/40 rounded-2xl flex flex-col justify-between shadow-[0_0_30px_rgba(0,240,255,0.25)] select-none">
      {/* Draggable Titlebar Header */}
      <div
        className="flex items-center justify-between border-b border-cyan-500/20 pb-2.5 cursor-move"
        style={{ WebkitAppRegion: 'drag' } as React.CSSProperties}
      >
        <div className="flex items-center gap-2">
          <Zap className="w-4 h-4 text-cyan-400 animate-pulse" />
          <span className="font-heading font-bold text-xs tracking-wider text-cyan-300">
            ULTRON HUD
          </span>
          <span
            className={`w-2 h-2 rounded-full ${
              isOnline ? 'bg-emerald-400' : 'bg-red-400'
            }`}
          />
        </div>

        {/* Non-draggable control buttons */}
        <div
          className="flex items-center gap-1.5"
          style={{ WebkitAppRegion: 'no-drag' } as React.CSSProperties}
        >
          <button
            onClick={handleOpenCommandCenter}
            title="Open Full Command Center"
            className="p-1 rounded hover:bg-cyan-950/80 text-cyan-400 hover:text-cyan-200 transition-colors"
          >
            <Maximize2 className="w-3.5 h-3.5" />
          </button>
          <button
            onClick={handleCloseHUD}
            title="Hide HUD"
            className="p-1 rounded hover:bg-red-950/80 text-slate-400 hover:text-red-300 transition-colors"
          >
            <X className="w-3.5 h-3.5" />
          </button>
        </div>
      </div>

      {/* Compact Robotic Avatar Visualizer */}
      <div className="flex-1 flex flex-col items-center justify-center my-2 relative">
        <div className="w-48 h-48 max-w-full">
          <AvatarContainer state={state.avatarState} />
        </div>
      </div>

      {/* State & Telemetry Card */}
      <div className="bg-slate-900/90 border border-cyan-500/20 rounded-xl p-3 font-code text-xs space-y-2">
        {/* Avatar Status Badge */}
        <div className="flex items-center justify-between">
          <span className="text-slate-400 text-[10px]">STATE:</span>
          <span
            className={`px-2 py-0.5 rounded text-[10px] font-bold border ${colorTheme.badge}`}
          >
            {state.avatarState}
          </span>
        </div>

        {/* Live Audio / VAD Activity */}
        <div className="flex items-center justify-between text-[11px]">
          <div className="flex items-center gap-1.5 text-slate-400">
            <Radio
              className={`w-3.5 h-3.5 ${
                state.audioState.vadActive ? 'text-emerald-400 animate-pulse' : 'text-slate-500'
              }`}
            />
            <span>VAD:</span>
          </div>
          <span
            className={
              state.audioState.vadActive ? 'text-emerald-400 font-bold' : 'text-slate-500'
            }
          >
            {state.audioState.vadActive ? 'SPEECH DETECTED' : 'SILENCE'}
          </span>
        </div>

        {/* System Telemetry (CPU & RAM) */}
        <div className="grid grid-cols-2 gap-2 pt-1 border-t border-slate-800 text-[10px]">
          <div className="flex items-center gap-1 text-slate-400">
            <Cpu className="w-3 h-3 text-cyan-400" />
            <span>CPU:</span>
            <span className="text-slate-200 font-semibold">
              {state.systemTelemetry.cpu_percent > 0
                ? `${state.systemTelemetry.cpu_percent}%`
                : '--'}
            </span>
          </div>
          <div className="flex items-center gap-1 text-slate-400">
            <Activity className="w-3 h-3 text-purple-400" />
            <span>RAM:</span>
            <span className="text-slate-200 font-semibold">
              {state.systemTelemetry.memory_used_gb > 0
                ? `${state.systemTelemetry.memory_used_gb.toFixed(1)}GB`
                : '--'}
            </span>
          </div>
        </div>
      </div>

      {/* Footer Navigation Action */}
      <div
        className="mt-2.5"
        style={{ WebkitAppRegion: 'no-drag' } as React.CSSProperties}
      >
        <button
          onClick={handleOpenCommandCenter}
          className="w-full py-1.5 bg-cyan-500/10 hover:bg-cyan-500/20 border border-cyan-500/40 hover:border-cyan-400 text-cyan-300 hover:text-cyan-100 font-code font-bold text-[11px] rounded-lg transition-all flex items-center justify-center gap-1.5 shadow-[0_0_12px_rgba(0,240,255,0.15)]"
        >
          <Maximize2 className="w-3 h-3" />
          <span>EXPAND COMMAND CENTER</span>
        </button>
      </div>
    </div>
  );
};
