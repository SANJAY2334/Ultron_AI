# PROJECT ULTRON - Backend Service

## Overview
This repository contains the backend service infrastructure for **PROJECT ULTRON** — Personal Artificial Intelligence Operating System.

Current Phase: **Phase 1 - ULTRON Kernel Foundation**

---

## Current Repository Structure

```
backend/
├── .env.example        # Environment variable configuration blueprint
├── pyproject.toml      # Build configuration and dependency manifest
├── README.md           # Repository documentation
└── app/
    ├── __init__.py     # Core application package initialization
    └── tests/
        ├── __init__.py
        └── test_infrastructure.py # Infrastructure verification tests
```

---

## Getting Started

### 1. Environment Setup
Copy the blueprint configuration file to create your local `.env`:
```bash
cp .env.example .env
```

### 2. Dependency Installation
Install dependencies in your local Python environment:
```bash
pip install -e ".[dev]"
```

### 3. Running Infrastructure Tests
Run pytest to verify environment setup and dependencies:
```bash
pytest
```
