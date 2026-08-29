import React from 'react';
import { ShieldCheck, ShieldAlert, CheckCircle, XCircle, AlertOctagon, Info } from 'lucide-react';
import type { SecuritySubsystemState, UltronAction } from '../state/UltronSystemState';

interface SecurityPanelProps {
  securityState: SecuritySubsystemState;
  dispatch: React.Dispatch<UltronAction>;
}

export const SecurityPanel: React.FC<SecurityPanelProps> = ({ securityState, dispatch }) => {
  const toggleCapability = (cap: string, currentlyGranted: boolean) => {
    dispatch({ 
      type: 'SECURITY_CAPABILITY_CHANGED', 
      capability: cap, 
      granted: !currentlyGranted 
    });
  };

  const handleConfirm = (requestId: string, confirmed: boolean) => {
    dispatch({
      type: 'SECURITY_CONFIRMATION_RESPONSE',
      confirmed,
      requestId
    });
  };

  const isLocked = securityState.status !== 'ONLINE';

  return (
    <div className="glass-panel p-5 flex flex-col h-full relative">
      {/* Pending Confirmation Modal Interlock */}
      {securityState.pendingConfirmation && (
        <div className="absolute inset-0 z-50 bg-slate-950/90 backdrop-blur-sm flex items-center justify-center p-4 rounded-xl border border-amber-500/50">
          <div className="bg-slate-900 border border-amber-500 rounded-lg p-5 max-w-sm w-full shadow-[0_0_30px_rgba(255,170,0,0.2)]">
            <div className="flex items-center gap-3 text-amber-400 mb-3 border-b border-amber-500/20 pb-2">
              <AlertOctagon className="w-6 h-6 animate-pulse" />
              <h3 className="font-heading font-bold text-lg">CONFIRMATION REQUIRED</h3>
            </div>
            
            <div className="mb-4 font-code text-sm text-slate-200">
              <p className="mb-2 font-bold text-amber-300">{securityState.pendingConfirmation.action}</p>
              <p className="text-slate-400 text-xs bg-slate-950 p-2 rounded border border-slate-800">
                {securityState.pendingConfirmation.reason}
              </p>
            </div>
            
            <div className="flex items-center gap-3 font-code text-sm">
              <button 
                onClick={() => handleConfirm(securityState.pendingConfirmation!.requestId, false)}
                className="flex-1 py-2 rounded bg-slate-800 hover:bg-slate-700 text-slate-300 transition-colors border border-slate-600"
              >
                CANCEL
              </button>
              <button 
                onClick={() => handleConfirm(securityState.pendingConfirmation!.requestId, true)}
                className="flex-1 py-2 rounded bg-amber-500 hover:bg-amber-400 text-black font-bold transition-colors shadow-[0_0_15px_rgba(255,170,0,0.4)]"
              >
                AUTHORIZE
              </button>
            </div>
            <p className="mt-3 text-[10px] text-amber-500/70 font-code text-center">
              SYSTEM REQUIRES EXPLICIT USER CONSENT
            </p>
          </div>
        </div>
      )}

      {/* Panel Header */}
      <div className="flex items-center justify-between mb-4 border-b border-cyan-500/20 pb-3">
        <div className="flex items-center gap-2">
          {securityState.status === 'ONLINE' ? (
            <ShieldCheck className="w-5 h-5 text-emerald-400" />
          ) : (
            <ShieldAlert className="w-5 h-5 text-red-400" />
          )}
          <h2 className="font-heading text-base font-bold text-cyan-300">
            ZERO-TRUST POLICY ENGINE & SAFETY INTERLOCK
          </h2>
        </div>
        <div className="flex items-center gap-1.5 text-xs font-code">
          <span className="text-slate-400">STATUS:</span>
          <span className={`font-bold ${securityState.status === 'ONLINE' ? 'text-emerald-400' : 'text-red-400'}`}>
            {securityState.status}
          </span>
        </div>
      </div>

      {/* Capability Matrix Section */}
      <div className="mb-4">
        <div className="text-xs font-code text-slate-400 mb-2 flex items-center justify-between">
          <span>GRANTED CAPABILITIES MATRIX:</span>
          <span className="text-[10px] text-slate-500">TOGGLE PERMISSIONS</span>
        </div>
        <div className="grid grid-cols-2 sm:grid-cols-3 gap-2 text-xs font-code">
          {Object.entries(securityState.grantedCapabilities || {}).map(([cap, granted]) => (
            <button
              key={cap}
              onClick={() => toggleCapability(cap, granted)}
              disabled={isLocked}
              className={`flex items-center justify-between px-3 py-2 rounded border transition-all ${
                isLocked ? 'opacity-50 cursor-not-allowed' : ''
              } ${
                granted
                  ? 'bg-emerald-950/60 text-emerald-300 border-emerald-500/40 shadow-[0_0_8px_rgba(0,255,106,0.15)]'
                  : 'bg-slate-950 text-slate-500 border-slate-800'
              }`}
            >
              <span className="truncate mr-2">{cap}</span>
              {granted ? (
                <CheckCircle className="w-3.5 h-3.5 flex-shrink-0 text-emerald-400" />
              ) : (
                <XCircle className="w-3.5 h-3.5 flex-shrink-0 text-slate-600" />
              )}
            </button>
          ))}
          {Object.keys(securityState.grantedCapabilities || {}).length === 0 && (
            <div className="col-span-full text-slate-500 italic p-2 bg-slate-900 rounded border border-slate-800 text-center">
              No capabilities registered
            </div>
          )}
        </div>
      </div>

      {/* Real-time Policy Decision Feed */}
      <div className="flex-1 bg-slate-950/80 rounded border border-slate-800 p-3 font-code text-xs overflow-y-auto min-h-[140px]">
        <div className="text-[10px] text-slate-500 uppercase mb-2 flex items-center gap-1.5">
          <Info className="w-3 h-3" />
          Policy Evaluation Stream
        </div>
        <div className="space-y-1.5">
          {!securityState.policyDecision ? (
            <div className="text-slate-600 italic px-1">Awaiting policy decisions...</div>
          ) : (
            (() => {
              const isAllow = securityState.policyDecision === 'ALLOW';
              const isReq = securityState.policyDecision === 'REQUIRES_CONFIRMATION';
              const colorClass = isAllow
                ? 'text-emerald-400 bg-emerald-950/40 border-emerald-500/30'
                : isReq
                ? 'text-amber-400 bg-amber-950/40 border-amber-500/30'
                : 'text-red-400 bg-red-950/40 border-red-500/30';

              return (
                <div className={`p-2 rounded border text-[11px] ${colorClass}`}>
                  <div className="flex items-center justify-between font-bold">
                    <span>[{securityState.policyDecision}]</span>
                  </div>
                  <div className="text-slate-300 mt-0.5">
                    {isAllow
                      ? 'Action authorized by Zero-Trust PolicyEngine.'
                      : isReq
                      ? 'Destructive action requires explicit user approval.'
                      : 'Action denied by PolicyEngine safety interlock.'}
                  </div>
                </div>
              );
            })()
          )}
        </div>
      </div>
    </div>
  );
};
