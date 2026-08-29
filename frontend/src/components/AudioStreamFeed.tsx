import React, { useEffect, useRef } from 'react';
import { Mic, MicOff, Radio, Sparkles, Activity } from 'lucide-react';
import type { AudioSubsystemState, TranscriptEntry, UltronAction } from '../state/UltronSystemState';

interface AudioStreamFeedProps {
  audioState: AudioSubsystemState;
  transcript: TranscriptEntry[];
  dispatch: React.Dispatch<UltronAction>;
}

export const AudioStreamFeed: React.FC<AudioStreamFeedProps> = ({ audioState, transcript, dispatch }) => {
  const canvasRef = useRef<HTMLCanvasElement | null>(null);

  const micActive = audioState.micActive;
  const vadActive = audioState.vadActive;
  const isUnavailable = audioState.status === 'UNAVAILABLE';

  // Audio Spectrum Waveform Canvas rendering
  useEffect(() => {
    if (isUnavailable) return;

    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    let animId: number;
    let phase = 0;

    const drawWaveform = () => {
      const w = canvas.width;
      const h = canvas.height;
      ctx.clearRect(0, 0, w, h);

      const numBars = 32;
      const barWidth = (w / numBars) - 3;

      for (let i = 0; i < numBars; i++) {
        const heightMultiplier = micActive
          ? vadActive
            ? Math.abs(Math.sin(phase + i * 0.2)) * 0.8 + 0.15
            : 0.1
          : 0.05;

        const barHeight = h * heightMultiplier;
        const x = i * (barWidth + 3);
        const y = h - barHeight;

        const gradient = ctx.createLinearGradient(0, h, 0, 0);
        gradient.addColorStop(0, 'rgba(0, 240, 255, 0.2)');
        gradient.addColorStop(1, vadActive ? '#00ff66' : '#00f0ff');

        ctx.fillStyle = gradient;
        ctx.fillRect(x, y, barWidth, barHeight);
      }

      phase += vadActive ? 0.2 : 0.05;
      animId = requestAnimationFrame(drawWaveform);
    };

    drawWaveform();

    return () => {
      cancelAnimationFrame(animId);
    };
  }, [micActive, vadActive, isUnavailable]);

  const filteredTranscript = transcript.filter(t => t.source === 'AUDIO' || t.source === 'USER');

  return (
    <div className="glass-panel p-5 flex flex-col h-full">
      {/* Panel Header */}
      <div className="flex items-center justify-between mb-4 border-b border-cyan-500/20 pb-3">
        <div className="flex items-center gap-2">
          <Mic className="w-5 h-5 text-cyan-400" />
          <h2 className="font-heading text-base font-bold text-cyan-300">
            AUDIO INTELLIGENCE STREAM
          </h2>
          <span className={`px-2 py-0.5 rounded text-[10px] font-code ml-2 ${
            audioState.status === 'ONLINE' ? 'bg-emerald-950 border border-emerald-500/40 text-emerald-400' :
            audioState.status === 'OFFLINE' ? 'bg-red-950 border border-red-500/40 text-red-400' :
            'bg-slate-900 border border-slate-700 text-slate-400'
          }`}>
            {audioState.status}
          </span>
        </div>
        <button
          onClick={() => dispatch({ type: 'AUDIO_MIC_TOGGLED', active: !micActive })}
          disabled={isUnavailable}
          className={`flex items-center gap-1.5 px-3 py-1 rounded text-xs font-code font-semibold transition-all ${
            isUnavailable ? 'bg-slate-900 text-slate-500 cursor-not-allowed border border-slate-800' :
            micActive
              ? 'bg-emerald-950/80 text-emerald-300 border border-emerald-500/40'
              : 'bg-red-950/80 text-red-300 border border-red-500/40'
          }`}
        >
          {micActive ? <Mic className="w-3.5 h-3.5 text-emerald-400" /> : <MicOff className="w-3.5 h-3.5 text-red-400" />}
          {micActive ? 'MICROPHONE ACTIVE' : 'MUTED'}
        </button>
      </div>

      {/* Waveform Canvas & Status Badges */}
      <div className="relative bg-slate-950 rounded-md p-4 border border-cyan-500/20 mb-4 flex-shrink-0">
        {isUnavailable ? (
          <div className="w-full h-[70px] flex items-center justify-center flex-col gap-2">
            <Radio className="w-6 h-6 text-slate-600" />
            <span className="text-slate-500 text-xs font-code">AUDIO SUBSYSTEM UNAVAILABLE</span>
          </div>
        ) : (
          <canvas ref={canvasRef} width={480} height={70} className="w-full h-[70px]" />
        )}

        <div className="flex items-center justify-between mt-3 text-xs font-code border-t border-slate-800 pt-2 flex-wrap gap-2">
          {/* VAD Status */}
          <div className="flex items-center gap-2">
            <Radio className={`w-3.5 h-3.5 ${vadActive ? 'text-emerald-400 animate-pulse' : 'text-slate-500'}`} />
            <span className="text-slate-400">VAD:</span>
            <span className={vadActive ? 'text-emerald-400 font-bold' : 'text-slate-500'}>
              {vadActive ? 'SPEECH DETECTED' : 'SILENCE'}
            </span>
          </div>

          {/* Wake Word */}
          <div className="flex items-center gap-2">
            <Sparkles className={`w-3.5 h-3.5 ${audioState.wakeWordTriggered ? 'text-amber-400 animate-bounce' : 'text-slate-500'}`} />
            <span className="text-slate-400">WAKE WORD:</span>
            <span className={audioState.wakeWordTriggered ? 'text-amber-300 font-bold' : 'text-slate-500'}>
              {audioState.wakeWordTriggered ? 'TRIGGERED' : 'LISTENING'}
            </span>
          </div>

          {/* TTS Status */}
          <div className="flex items-center gap-2">
            <Activity className="w-3.5 h-3.5 text-cyan-500" />
            <span className="text-slate-400">TTS:</span>
            <span className="text-cyan-400">{audioState.ttsStatus}</span>
          </div>

          {/* Latency */}
          {audioState.latencyMs > 0 && (
            <div className="text-slate-500">
              {audioState.latencyMs}ms
            </div>
          )}
        </div>
      </div>

      {/* STT Confidence */}
      {audioState.currentTranscript && (
        <div className="bg-slate-950/80 rounded border border-cyan-500/20 p-3 mb-4 font-code text-xs">
          <div className="text-slate-400 text-[10px] mb-1">CURRENT TRANSCRIPT</div>
          <div className="text-cyan-300 font-semibold">"{audioState.currentTranscript}"</div>
          <div className="text-slate-500 text-[10px] mt-1">
            STT Confidence: {(audioState.sttConfidence * 100).toFixed(0)}%
          </div>
        </div>
      )}

      {/* Transcript Log */}
      <div className="flex-1 bg-slate-950/80 rounded border border-slate-800 p-3 font-code text-xs overflow-y-auto">
        <div className="text-[10px] text-slate-500 mb-2">TRANSCRIPT FEED</div>
        <div className="space-y-1">
          {filteredTranscript.length === 0 && (
            <div className="text-slate-600 italic">No transcripts yet...</div>
          )}
          {filteredTranscript.map((entry) => (
            <div key={entry.id} className="text-slate-300 py-1 border-b border-slate-900/60 last:border-0">
              <span className="text-slate-500 mr-2">[{new Date(entry.timestamp).toLocaleTimeString()}]</span>
              <span className={entry.source === 'USER' ? 'text-cyan-300' : 'text-emerald-300'}>
                {entry.content}
              </span>
            </div>
          ))}
        </div>
      </div>
    </div>
  );
};
