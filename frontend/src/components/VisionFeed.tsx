import React, { useEffect, useRef } from 'react';
import { Camera, ShieldCheck, Crosshair } from 'lucide-react';
import type { VisionSubsystemState } from '../state/UltronSystemState';
import type { ObjectDetection, ObjectTrack, SceneSummary } from '../types';

interface VisionFeedProps {
  visionState: VisionSubsystemState;
  activeObjects: ObjectDetection[];
  activeTracks: ObjectTrack[];
  sceneSummary: SceneSummary | null;
}

export const VisionFeed: React.FC<VisionFeedProps> = ({ visionState, activeObjects, activeTracks, sceneSummary }) => {
  const canvasRef = useRef<HTMLCanvasElement | null>(null);
  
  const isOnline = visionState.status === 'ONLINE' || visionState.cameraOnline;

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    let animationFrameId: number;

    const render = () => {
      const w = canvas.width;
      const h = canvas.height;

      // Dark sci-fi background
      ctx.fillStyle = '#060a14';
      ctx.fillRect(0, 0, w, h);

      if (!isOnline) {
        ctx.fillStyle = '#ff0033';
        ctx.font = 'bold 20px "JetBrains Mono", monospace';
        ctx.textAlign = 'center';
        ctx.fillText('VISION SUBSYSTEM UNAVAILABLE/OFFLINE', w / 2, h / 2);
        animationFrameId = requestAnimationFrame(render);
        return;
      }

      // Grid lines
      ctx.strokeStyle = 'rgba(0, 240, 255, 0.05)';
      ctx.lineWidth = 1;
      const step = 40;
      for (let x = 0; x < w; x += step) {
        ctx.beginPath();
        ctx.moveTo(x, 0);
        ctx.lineTo(x, h);
        ctx.stroke();
      }
      for (let y = 0; y < h; y += step) {
        ctx.beginPath();
        ctx.moveTo(0, y);
        ctx.lineTo(w, y);
        ctx.stroke();
      }

      ctx.textAlign = 'left';
      // Draw bounding box targets
      activeObjects.forEach((det, idx) => {
        const bbox = det.bounding_box;
        const x = bbox.x * w;
        const y = bbox.y * h;
        const bw = bbox.width * w;
        const bh = bbox.height * h;

        const isPerson = det.label === 'person';
        const color = isPerson ? '#00f0ff' : '#9d00ff';

        // Draw Bounding Box Corner Brackets
        ctx.strokeStyle = color;
        ctx.lineWidth = 2;
        const cornerSize = 12;

        ctx.beginPath(); ctx.moveTo(x, y + cornerSize); ctx.lineTo(x, y); ctx.lineTo(x + cornerSize, y); ctx.stroke();
        ctx.beginPath(); ctx.moveTo(x + bw - cornerSize, y); ctx.lineTo(x + bw, y); ctx.lineTo(x + bw, y + cornerSize); ctx.stroke();
        ctx.beginPath(); ctx.moveTo(x, y + bh - cornerSize); ctx.lineTo(x, y + bh); ctx.lineTo(x + cornerSize, y + bh); ctx.stroke();
        ctx.beginPath(); ctx.moveTo(x + bw - cornerSize, y + bh); ctx.lineTo(x + bw, y + bh); ctx.lineTo(x + bw, y + bh - cornerSize); ctx.stroke();

        ctx.fillStyle = isPerson ? 'rgba(0, 240, 255, 0.06)' : 'rgba(157, 0, 255, 0.06)';
        ctx.fillRect(x, y, bw, bh);

        const track = activeTracks[idx];
        const trackId = track ? track.track_id : `track_${idx}`;
        // person != identity, so we just use the class label
        const labelText = `[ ${det.label.toUpperCase()} | ${trackId} | ${(det.confidence * 100).toFixed(0)}% ]`;

        ctx.fillStyle = 'rgba(7, 10, 18, 0.85)';
        ctx.fillRect(x, y - 24, ctx.measureText(labelText).width + 16, 20);

        ctx.fillStyle = color;
        ctx.font = '11px "JetBrains Mono", monospace';
        ctx.fillText(labelText, x + 8, y - 10);
      });

      // Reticle in center
      ctx.strokeStyle = 'rgba(0, 240, 255, 0.2)';
      ctx.beginPath();
      ctx.arc(w / 2, h / 2, 20, 0, Math.PI * 2);
      ctx.stroke();

      animationFrameId = requestAnimationFrame(render);
    };

    render();

    return () => {
      cancelAnimationFrame(animationFrameId);
    };
  }, [isOnline, activeObjects, activeTracks]);

  return (
    <div className="glass-panel p-5 flex flex-col h-full">
      <div className="flex items-center justify-between mb-4 border-b border-cyan-500/20 pb-3">
        <div className="flex items-center gap-2">
          <Camera className="w-5 h-5 text-cyan-400" />
          <h2 className="font-heading text-base font-bold text-cyan-300">
            VISION PERCEPTION FEED
          </h2>
        </div>
      </div>

      <div className="relative aspect-video rounded-md overflow-hidden border border-cyan-500/30 bg-slate-950 mb-4 flex-1">
        <canvas ref={canvasRef} width={640} height={360} className="w-full h-full object-cover" />
        <div className="scanline-overlay" />

        <div className="absolute top-3 left-3 flex items-center gap-2 bg-slate-950/80 px-2.5 py-1 rounded border border-cyan-500/30 text-[11px] font-code text-cyan-300">
          <Crosshair className={`w-3.5 h-3.5 text-cyan-400 ${visionState.cameraOnline ? 'animate-spin' : ''}`} />
          <span>CAM_01: {visionState.cameraOnline ? 'ONLINE' : 'OFFLINE'}</span>
        </div>

        <div className="absolute bottom-3 left-3 flex items-center gap-1.5 bg-emerald-950/90 px-2.5 py-1 rounded border border-emerald-500/40 text-[11px] font-code text-emerald-300">
          <ShieldCheck className="w-3.5 h-3.5 text-emerald-400" />
          <span>ANONYMITY ENFORCED (extra="forbid")</span>
        </div>

        <div className="absolute bottom-3 right-3 bg-slate-900/90 px-2 py-1 rounded border border-slate-700 text-[10px] font-code text-slate-400">
          PRIVACY: EPHEMERAL (NO DISK PERSISTENCE)
        </div>
      </div>

      <div className="grid grid-cols-2 sm:grid-cols-4 gap-3 font-code text-xs">
        <div className="bg-slate-950/60 p-2.5 rounded border border-slate-800">
          <div className="text-slate-400 text-[10px] uppercase">Scene State</div>
          <div className="font-bold text-cyan-400 mt-0.5">{sceneSummary?.scene_state ?? 'STABLE'}</div>
        </div>
        <div className="bg-slate-950/60 p-2.5 rounded border border-slate-800">
          <div className="text-slate-400 text-[10px] uppercase">Motion Level</div>
          <div className="font-bold text-purple-400 mt-0.5">{sceneSummary?.motion_level ?? 'LOW'}</div>
        </div>
        <div className="bg-slate-950/60 p-2.5 rounded border border-slate-800">
          <div className="text-slate-400 text-[10px] uppercase">Objects Detected</div>
          <div className="font-bold text-emerald-400 mt-0.5">
            {activeObjects.length} ({activeTracks.length} Tracks)
          </div>
        </div>
        <div className="bg-slate-950/60 p-2.5 rounded border border-slate-800">
          <div className="text-slate-400 text-[10px] uppercase">Dominant Activity</div>
          <div className="font-bold text-amber-400 mt-0.5">{sceneSummary?.dominant_activity ?? 'STATIONARY'}</div>
        </div>
      </div>
    </div>
  );
};
