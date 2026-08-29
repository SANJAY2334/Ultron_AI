import React, { useState, useRef } from 'react';
import { Terminal, Send, Bot, Eye, CheckCircle2, AlertTriangle, Clock } from 'lucide-react';
import type { PlannerSubsystemState, PlannerEventEntry, UltronAction } from '../state/UltronSystemState';
import type { VisionPlannerContext } from '../types';
import { connectPlannerStream } from '../services/api';
import { VisionContextPanel } from './VisionContextPanel';

interface PlannerPanelProps {
  plannerState: PlannerSubsystemState;
  plannerEvents: PlannerEventEntry[];
  visionContext: VisionPlannerContext | null;
  dispatch: React.Dispatch<UltronAction>;
}

export const PlannerPanel: React.FC<PlannerPanelProps> = ({ plannerState, plannerEvents, visionContext, dispatch }) => {
  const [prompt, setPrompt] = useState<string>('');
  const abortRef = useRef<AbortController | null>(null);

  const isOffline = plannerState.status === 'OFFLINE';
  const isExecuting = plannerState.isExecuting;

  const handleExecute = () => {
    if (!prompt.trim() || isExecuting || isOffline) return;

    const cmd = prompt;
    setPrompt('');

    dispatch({ type: 'PLANNER_EXECUTION_STARTED' });

    // Cancel any previous SSE stream
    if (abortRef.current) abortRef.current.abort();

    abortRef.current = connectPlannerStream(
      cmd,
      // onEvent
      (sseEvent) => {
        // Map SSE event types to PlannerStreamEvent event_types
        let eventType: 'THINKING' | 'TOOL_STARTED' | 'TOOL_COMPLETED' | 'PARTIAL_RESPONSE' | 'COMPLETED' | 'ERROR';
        let payload = '';
        const reqId = (sseEvent.data.planner_id as string) ?? `req_${Date.now()}`;

        switch (sseEvent.eventType) {
          case 'start':
            eventType = 'THINKING';
            payload = 'Planner pipeline started...';
            break;
          case 'tool_status':
            eventType = (sseEvent.data.status as string) === 'completed' ? 'TOOL_COMPLETED' : 'TOOL_STARTED';
            payload = `Tool: ${sseEvent.data.tool_name ?? 'unknown'} (${sseEvent.data.execution_time_ms ?? 0}ms)`;
            break;
          case 'token':
            eventType = 'PARTIAL_RESPONSE';
            payload = (sseEvent.data.chunk as string) ?? '';
            break;
          case 'completion':
            eventType = 'COMPLETED';
            payload = (sseEvent.data.response as string) ?? 'Completed';
            break;
          case 'error':
            eventType = 'ERROR';
            payload = (sseEvent.data.message as string) ?? 'Unknown error';
            break;
          default:
            return; // Skip unknown event types
        }

        dispatch({
          type: 'PLANNER_EVENT',
          event: {
            event_type: eventType,
            payload,
            request_id: reqId,
            timestamp: new Date().toISOString(),
          },
        });
      },
      // onError
      (errorMsg) => {
        dispatch({
          type: 'PLANNER_EVENT',
          event: {
            event_type: 'ERROR',
            payload: errorMsg,
            request_id: `err_${Date.now()}`,
            timestamp: new Date().toISOString(),
          },
        });
      },
      // onComplete
      () => {
        dispatch({ type: 'PLANNER_EXECUTION_ENDED' });
        abortRef.current = null;
      },
    );
  };

  return (
    <div className="glass-panel p-5 flex flex-col h-full">
      {/* Panel Header */}
      <div className="flex items-center justify-between mb-4 border-b border-cyan-500/20 pb-3">
        <div className="flex items-center gap-2">
          <Bot className="w-5 h-5 text-cyan-400" />
          <h2 className="font-heading text-base font-bold text-cyan-300">
            AUTONOMOUS PLANNER GATEWAY
          </h2>
          <span className={`px-2 py-0.5 rounded text-[10px] font-code ${
            plannerState.status === 'ONLINE' ? 'bg-emerald-950 border border-emerald-500/40 text-emerald-400' :
            'bg-red-950 border border-red-500/40 text-red-400'
          }`}>
            {plannerState.status}
          </span>
        </div>
        <div className="flex items-center gap-2 text-xs font-code text-slate-400">
          <Eye className="w-3.5 h-3.5 text-purple-400" />
          <span>{visionContext ? 'VISUAL CONTEXT INJECTED' : 'NO VISUAL CONTEXT'}</span>
        </div>
      </div>

      {/* Vision Context */}
      <div className="mb-4">
        <VisionContextPanel context={visionContext} />
      </div>

      {/* Execution Event Log */}
      <div className="flex-1 bg-slate-950 rounded border border-slate-800 p-4 font-code text-xs overflow-y-auto mb-4 space-y-2 min-h-[160px]">
        {plannerEvents.length === 0 && (
          <div className="text-slate-600 italic">No planner events yet...</div>
        )}
        {plannerEvents.map((evt) => {
          let badgeColor = 'bg-cyan-950 text-cyan-300 border-cyan-500/40';
          let Icon = Clock;

          if (evt.eventType === 'TOOL_STARTED') {
            badgeColor = 'bg-purple-950 text-purple-300 border-purple-500/40';
            Icon = Terminal;
          } else if (evt.eventType === 'TOOL_COMPLETED' || evt.eventType === 'COMPLETED') {
            badgeColor = 'bg-emerald-950 text-emerald-300 border-emerald-500/40';
            Icon = CheckCircle2;
          } else if (evt.eventType === 'ERROR') {
            badgeColor = 'bg-red-950 text-red-300 border-red-500/40';
            Icon = AlertTriangle;
          } else if (evt.eventType === 'THINKING') {
            badgeColor = 'bg-cyan-950 text-cyan-300 border-cyan-500/40';
            Icon = Bot;
          }

          return (
            <div key={evt.id} className="flex flex-col gap-1 pb-2 border-b border-slate-900 last:border-0">
              <div className="flex items-center justify-between">
                <span className={`inline-flex items-center gap-1 px-2 py-0.5 rounded text-[10px] font-bold border ${badgeColor}`}>
                  <Icon className="w-3 h-3" />
                  {evt.eventType}
                </span>
                <span className="text-[10px] text-slate-500">{new Date(evt.timestamp).toLocaleTimeString()}</span>
              </div>
              <p className="text-slate-200 pl-1">{evt.payload}</p>
            </div>
          );
        })}
      </div>

      {/* Command Input */}
      <form
        onSubmit={(e) => {
          e.preventDefault();
          handleExecute();
        }}
        className="flex items-center gap-2"
      >
        <div className="relative flex-1">
          <input
            type="text"
            value={prompt}
            onChange={(e) => setPrompt(e.target.value)}
            placeholder={isOffline ? 'PLANNER OFFLINE — backend unreachable' : "Type command (e.g. 'Get system info')..."}
            className="w-full bg-slate-950 border border-slate-800 focus:border-cyan-500/60 rounded px-3 py-2 text-xs font-code text-slate-100 placeholder:text-slate-500 focus:outline-none transition-all disabled:opacity-50"
            disabled={isExecuting || isOffline}
          />
        </div>
        <button
          type="submit"
          disabled={isExecuting || !prompt.trim() || isOffline}
          className="flex items-center gap-1.5 px-4 py-2 bg-cyan-500 hover:bg-cyan-400 disabled:opacity-50 text-black font-code font-bold text-xs rounded transition-all shadow-[0_0_12px_rgba(0,240,255,0.4)]"
        >
          <Send className="w-3.5 h-3.5" />
          <span>EXECUTE</span>
        </button>
      </form>
    </div>
  );
};
