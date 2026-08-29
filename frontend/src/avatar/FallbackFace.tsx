import React from 'react';
import type { AvatarState } from './types';
import { getAvatarColor } from './stateMapper';

interface FallbackFaceProps {
  state: AvatarState;
}

export const FallbackFace: React.FC<FallbackFaceProps> = ({ state }) => {
  const color = getAvatarColor(state);

  return (
    <div className="w-full aspect-square max-w-[320px] mx-auto bg-slate-950 rounded-lg p-6 border border-slate-800 flex flex-col items-center justify-between text-center font-code">
      <div className="text-xs text-slate-400 font-bold uppercase tracking-wider">
        ULTRON ROBOTIC FACE [CSS FALLBACK]
      </div>

      {/* Simplified CSS Robotic Head Geometry */}
      <div
        className="w-32 h-36 rounded-2xl border-2 flex flex-col items-center justify-around p-3 relative shadow-lg"
        style={{ borderColor: color.primary, backgroundColor: '#090d18' }}
      >
        {/* Forehead Optic */}
        <div className="w-6 h-2 rounded-full" style={{ backgroundColor: color.primary }} />

        {/* Eyes */}
        <div className="flex items-center justify-between w-full px-2">
          <div
            className="w-7 h-7 rounded-full border flex items-center justify-center animate-pulse"
            style={{ borderColor: color.primary, backgroundColor: 'rgba(0,0,0,0.8)' }}
          >
            <div className="w-3 h-3 rounded-full" style={{ backgroundColor: color.primary }} />
          </div>
          <div
            className="w-7 h-7 rounded-full border flex items-center justify-center animate-pulse"
            style={{ borderColor: color.primary, backgroundColor: 'rgba(0,0,0,0.8)' }}
          >
            <div className="w-3 h-3 rounded-full" style={{ backgroundColor: color.primary }} />
          </div>
        </div>

        {/* Mouth Equalizer Bar */}
        <div className="w-16 h-2 rounded" style={{ backgroundColor: color.primary }} />
      </div>

      <div className="text-xs font-bold px-3 py-1 rounded bg-slate-900 border border-slate-700" style={{ color: color.primary }}>
        STATE: {state}
      </div>
    </div>
  );
};
