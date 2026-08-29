/* ULTRON Vision Context Panel — Sanitized Display (Phase 4G.3)
 *
 * Displays the sanitized VisionPlannerContext passed from the unified state.
 * Shows only aggregate scene metrics — no raw images, no internal reasoning.
 *
 * SECURITY INVARIANT: person ≠ identity. No biometric data displayed.
 */

import React from 'react';
import { Eye, Activity } from 'lucide-react';
import type { VisionPlannerContext } from '../types';

interface VisionContextPanelProps {
  context: VisionPlannerContext | null;
}

export const VisionContextPanel: React.FC<VisionContextPanelProps> = ({ context }) => {
  if (!context) {
    return (
      <div className="bg-slate-950/80 rounded border border-slate-800 p-3 text-xs font-code">
        <div className="flex items-center gap-1.5 text-slate-500">
          <Eye className="w-3.5 h-3.5" />
          <span>VISUAL CONTEXT: UNAVAILABLE</span>
        </div>
      </div>
    );
  }

  return (
    <div className="bg-slate-950/80 rounded border border-purple-500/30 p-3 text-xs font-code">
      <div className="text-purple-300 font-bold mb-2 flex items-center gap-1.5">
        <Activity className="w-3.5 h-3.5 text-purple-400" />
        <span>VISUAL CONTEXT</span>
      </div>
      <div className="grid grid-cols-2 gap-x-4 gap-y-1 text-slate-300">
        <div>
          Scene: <span className="text-cyan-300 font-semibold">{context.scene_state}</span>
        </div>
        <div>
          Motion: <span className="text-purple-300 font-semibold">{context.motion_level}</span>
        </div>
        <div>
          Objects: <span className="text-emerald-300 font-semibold">{context.object_count}</span>
        </div>
        <div>
          Active Tracks: <span className="text-amber-300 font-semibold">{context.active_track_count}</span>
        </div>
        {Object.entries(context.class_counts).map(([cls, count]) => (
          <div key={cls}>
            {cls}: <span className="text-cyan-300 font-semibold">{count}</span>
          </div>
        ))}
        <div className="col-span-2 mt-1">
          Dominant Activity: <span className="text-amber-300 font-semibold">{context.dominant_activity}</span>
        </div>
      </div>
    </div>
  );
};
