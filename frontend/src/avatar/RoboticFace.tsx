import React, { useEffect, useState } from 'react';
import type { AvatarState } from './types';
import { getAvatarColor } from './stateMapper';

interface RoboticFaceProps {
  state: AvatarState;
  compact?: boolean;
}

export const RoboticFace: React.FC<RoboticFaceProps> = ({ state, compact = false }) => {
  const [pulse, setPulse] = useState<number>(0);
  const color = getAvatarColor(state);

  // Low CPU idle animation pulse tick
  useEffect(() => {
    const interval = setInterval(() => {
      setPulse((p) => (p + 1) % 100);
    }, 50);

    return () => clearInterval(interval);
  }, []);

  const eyeScale = state === 'LISTENING' ? 1.25 : state === 'THINKING' ? 0.9 : 1.0;
  const mouthActive = state === 'SPEAKING' || state === 'SPEECH_DETECTED';

  if (compact) {
    return (
      <div className="relative w-8 h-8 flex items-center justify-center">
        <svg viewBox="0 0 100 100" className="w-full h-full">
          {/* Outer Cyber Shield */}
          <polygon points="50,5 90,25 90,75 50,95 10,75 10,25" fill="#0d1324" stroke={color.primary} strokeWidth="4" />
          {/* Eyes */}
          <circle cx="35" cy="45" r="8" fill={color.primary} />
          <circle cx="65" cy="45" r="8" fill={color.primary} />
          {/* Mouth line */}
          <line x1="35" y1="70" x2="65" y2="70" stroke={color.primary} strokeWidth="3" />
        </svg>
        <div
          className="absolute inset-0 rounded-full opacity-30 animate-ping"
          style={{ backgroundColor: color.primary }}
        />
      </div>
    );
  }

  return (
    <div className="relative w-full aspect-square max-w-[380px] mx-auto flex items-center justify-center p-4">
      {/* Outer Hexagonal Glowing Cyber Visor Frame */}
      <div className="relative w-full h-full flex items-center justify-center">
        <svg viewBox="0 0 400 400" className="w-full h-full drop-shadow-[0_0_25px_rgba(0,240,255,0.3)]">
          <defs>
            {/* Metal Gradients */}
            <linearGradient id="metalPlate" x1="0%" y1="0%" x2="100%" y2="100%">
              <stop offset="0%" stopColor="#1a233a" />
              <stop offset="50%" stopColor="#0c1222" />
              <stop offset="100%" stopColor="#060a14" />
            </linearGradient>

            <linearGradient id="eyeGlow" x1="0%" y1="0%" x2="100%" y2="100%">
              <stop offset="0%" stopColor={color.primary} />
              <stop offset="100%" stopColor="#000000" />
            </linearGradient>

            <filter id="glow" x="-20%" y="-20%" width="140%" height="140%">
              <feGaussianBlur stdDeviation="6" result="blur" />
              <feComposite in="SourceGraphic" in2="blur" operator="over" />
            </filter>
          </defs>

          {/* Background Sci-Fi Target Grid Circle */}
          <circle cx="200" cy="200" r="185" fill="none" stroke="rgba(0, 240, 255, 0.12)" strokeWidth="1" strokeDasharray="4,6" />
          <circle cx="200" cy="200" r="165" fill="none" stroke="rgba(157, 0, 255, 0.08)" strokeWidth="1" />

          {/* Robotic Cranial Side Armor Plates */}
          <polygon points="40,120 70,60 140,40 70,140" fill="url(#metalPlate)" stroke="rgba(0, 240, 255, 0.3)" strokeWidth="1.5" />
          <polygon points="360,120 330,60 260,40 330,140" fill="url(#metalPlate)" stroke="rgba(0, 240, 255, 0.3)" strokeWidth="1.5" />

          {/* Main Mechanical Visor Faceplate */}
          <polygon
            points="100,70 300,70 340,150 300,320 200,360 100,320 60,150"
            fill="url(#metalPlate)"
            stroke={color.primary}
            strokeWidth="2.5"
            filter="url(#glow)"
          />

          {/* Forehead Optic Sensor Gem */}
          <polygon points="180,90 220,90 210,110 190,110" fill={color.primary} opacity="0.9" />
          <line x1="150" y1="100" x2="250" y2="100" stroke={color.primary} strokeWidth="1" opacity="0.4" />

          {/* Temporal Panel Seams & Internal Illumination Vents */}
          <line x1="80" y1="170" x2="130" y2="170" stroke={color.primary} strokeWidth="1.5" opacity="0.6" />
          <line x1="320" y1="170" x2="270" y2="170" stroke={color.primary} strokeWidth="1.5" opacity="0.6" />

          {/* LEFT EYE APERTURE */}
          <g transform={`translate(135, 175) scale(${eyeScale})`}>
            {/* Eye Housing */}
            <polygon points="-35,-20 35,-20 45,15 0,30 -45,15" fill="#050811" stroke={color.primary} strokeWidth="2" />
            {/* Glowing Pupil Iris */}
            <circle cx="0" cy="0" r="14" fill={color.primary} filter="url(#glow)" />
            <circle cx="0" cy="0" r="6" fill="#ffffff" />
            {/* Rotating Reticle Ring */}
            <circle
              cx="0"
              cy="0"
              r="22"
              fill="none"
              stroke={color.primary}
              strokeWidth="1.5"
              strokeDasharray="8,4"
              transform={`rotate(${pulse * 3.6})`}
            />
          </g>

          {/* RIGHT EYE APERTURE */}
          <g transform={`translate(265, 175) scale(${eyeScale})`}>
            {/* Eye Housing */}
            <polygon points="-35,-20 35,-20 45,15 0,30 -45,15" fill="#050811" stroke={color.primary} strokeWidth="2" />
            {/* Glowing Pupil Iris */}
            <circle cx="0" cy="0" r="14" fill={color.primary} filter="url(#glow)" />
            <circle cx="0" cy="0" r="6" fill="#ffffff" />
            {/* Rotating Reticle Ring */}
            <circle
              cx="0"
              cy="0"
              r="22"
              fill="none"
              stroke={color.primary}
              strokeWidth="1.5"
              strokeDasharray="8,4"
              transform={`rotate(${-pulse * 3.6})`}
            />
          </g>

          {/* Cheek & Jaw Seam Lines */}
          <path d="M 120,220 L 160,260 L 200,270 L 240,260 L 280,220" fill="none" stroke="rgba(0, 240, 255, 0.3)" strokeWidth="1.5" />

          {/* VOICE FREQUENCY EQUALIZER MOUTH GRID */}
          <g transform="translate(140, 290)">
            {[0, 1, 2, 3, 4, 5, 6, 7, 8, 9].map((i) => {
              const h = mouthActive
                ? Math.abs(Math.sin((pulse + i * 15) * 0.1)) * 24 + 4
                : 4;
              return (
                <rect
                  key={i}
                  x={i * 12}
                  y={15 - h / 2}
                  width="7"
                  height={h}
                  rx="2"
                  fill={color.primary}
                  opacity={mouthActive ? 0.9 : 0.4}
                />
              );
            })}
          </g>

          {/* Chin Armor Vent Plate */}
          <polygon points="170,335 230,335 210,352 190,352" fill="#10172a" stroke={color.primary} strokeWidth="1" />
        </svg>

        {/* Scanlines overlay */}
        <div className="scanline-overlay" />
      </div>
    </div>
  );
};
