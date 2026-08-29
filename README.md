# ULTRON — Autonomous Multimodal Intelligence & Robotic Command Center

ULTRON is an enterprise-grade, Zero-Trust autonomous AI system combining real-time audio intelligence, spatial vision intelligence, deterministic autonomous planning, a 4-tier hybrid memory matrix, and a native Windows desktop Command Center with robotic avatar visualization.

---

## 🏛️ System Architecture

```
                                 ULTRON
                                    │
          ┌─────────────────────────┴─────────────────────────┐
          │                                                   │
   Native Desktop Shell (Electron)                     Authoritative Backend (FastAPI)
          │                                                   │
   Presentation & Desktop Lifecycle                     Zero-Trust Execution Boundary
          │                                                   │
   ├── Command Center Window (1440x900)                 ├── PolicyEngine
   ├── HUD Overlay Window (360x440 Frameless)           ├── SafetyInterlock
   ├── System Tray Singleton (Reticle Icon)             ├── ToolExecutor
   ├── Global Accelerator (Ctrl+Shift+U)                ├── Working & Episodic Memory
   ├── Sanitized OS Notifications                       └── Multimodal Context (Audio/Vision)
   └── Bounded Reconnect Backoff
```

### Core Security Invariants
- **Avatar ≠ Authority**: The avatar and HUD are read-only presentation projections of backend state.
- **Desktop Shell ≠ ToolExecutor**: The Electron shell possesses zero filesystem modification, tool execution, or process elevation authority.
- **Frontend ≠ PolicyEngine**: All safety interlock authorizations, capability evaluations, and tool policies remain 100% authoritative within the Python backend kernel.
- **Strict IPC Whitelist**: Strictly 8 validated channels exposed via `contextBridge.exposeInMainWorld`.
- **Zero Sensitive Data Leakage**: Multi-vector sanitization redacts JWTs, API keys, passwords, authorization headers, biometric vectors, raw frames, system prompts, and `<thought>` internal reasoning blocks.

---

## 🚀 Completed Phases (Phases 1 → 4G.4)

| Phase | Description | Status |
| :--- | :--- | :---: |
| **Phase 1** | Foundation & Zero-Trust Architecture | ✅ Complete |
| **Phase 2** | AI Core & LangGraph Planning Engine | ✅ Complete |
| **Phase 3** | Desktop Automation & OS Tool Execution | ✅ Complete |
| **Phase 4A** | Context Engine & Token Budgeting | ✅ Complete |
| **Phase 4B** | 4-Tier Hybrid Memory Matrix (Working, Episodic, Semantic, Graph) | ✅ Complete |
| **Phase 4C** | Audio Intelligence (VAD, Wake-Word, STT, TTS) | ✅ Complete |
| **Phase 4D** | Autonomous Voice Planner Gateway & Streaming | ✅ Complete |
| **Phase 4E** | Safety Interlock & Destructive Action Gating | ✅ Complete |
| **Phase 4F** | Vision Intelligence (Detection, Tracking, Scene Understanding) | ✅ Complete |
| **Phase 4G.1** | Live Multimodal Command Center Web UI | ✅ Complete |
| **Phase 4G.2** | Robotic Avatar Engine & State Mapper | ✅ Complete |
| **Phase 4G.3** | Live Multimodal Interface Integration (Zero-Demo Baseline) | ✅ Complete |
| **Phase 4G.4** | Native Desktop ULTRON (Electron, HUD, Tray, Hotkeys, Reconnect) | ✅ Complete |

---

## 🧪 Verification & Test Suite

- **Frontend Tests**: `94 passed, 0 failed` across 10 Vitest test suites.
- **Backend Tests**: `360 passed, 4 skipped, 0 failed` across 35 pytest test suites.
- **Linter & Type Checker**: Ruff clean, MyPy clean across 173 source files.
- **Production Build**: Clean Vite production bundle + compiled Electron main/preload.

---

## 🛠️ Quickstart Guide

### Prerequisites
- Python 3.11+ / Python 3.14
- Node.js v20+ / v24+
- npm v10+ / v11+

### 1. Backend Setup
```bash
cd backend
python -m venv .venv
# On Windows:
.venv\Scripts\activate
# On Linux/macOS:
source .venv/bin/activate

pip install -e .
python -m uvicorn app.main:app --host 127.0.0.1 --port 8000
```

### 2. Frontend & Native Desktop Setup
```bash
cd frontend
npm install

# Run web development server
npm run dev

# Run native desktop development (Vite + Electron)
npm run dev:desktop

# Run full test suite
npm test

# Build production desktop application
npm run build:desktop

# Package standalone Windows x64 binary
npm run package:dir
```

---

## 📦 Project Structure

```
Ultron/
├── backend/
│   ├── app/
│   │   ├── ai/            # Planner, AI router, tool executor, tool registry
│   │   ├── api/           # FastAPI routers and endpoints
│   │   ├── audio/         # Audio pipeline, VAD, wake-word, STT, TTS
│   │   ├── core/          # DI container, config, circuit breaker, logging
│   │   ├── desktop/       # Desktop automation models and action contracts
│   │   ├── memory/        # 4-tier memory matrix (working, episodic, semantic, graph)
│   │   ├── security/      # PolicyEngine, SafetyInterlock, PathPolicy
│   │   ├── vision/        # Detection, tracking, scene analysis, spatial context
│   │   └── main.py        # Application entrypoint
│   ├── pyproject.toml
│   └── app/tests/         # 360+ comprehensive unit & integration tests
├── frontend/
│   ├── electron/          # Main process, preload bridge, HUD, Tray, Hotkey, Notifications
│   ├── src/
│   │   ├── avatar/        # Robotic face SVG renderer and state mapper
│   │   ├── components/    # Header, VisionFeed, AudioFeed, PlannerPanel, HUDOverlay
│   │   ├── state/         # Unified system state reducer and actions
│   │   └── App.tsx        # Dual-view presentation root (Command Center & HUD)
│   ├── package.json
│   ├── vite.config.ts
│   └── electron-builder.json
└── README.md
```

---

## 📜 License
Proprietary & Confidential. All rights reserved.
