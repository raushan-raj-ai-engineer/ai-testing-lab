# Session 25 — Senior AI Automation / AI SDET Interview Cheat Sheet

## Core project story
I built an AI testing lab covering LLM and RAG evaluation, MCP tool contracts,
autonomous browser-agent decisions and trajectories, multi-agent handoffs,
deterministic guardrails, observability, reliability, Agentic RAG, AI API quality
and CI release gates.

## High-value distinctions
- **Deterministic vs probabilistic:** use known ground truth whenever available; semantic metrics supplement it.
- **Tool success vs task completion:** successful MCP execution does not prove the business goal completed.
- **Decision retry vs execution retry:** invalid reasoning needs re-planning; transient infrastructure failures may retry the same valid action.
- **Multi-agent:** test routing, handoff contracts, context preservation, dependencies, failure propagation and loops.
- **Safety:** guard input, handoff and tool boundaries; use least privilege.
- **Observability:** a functional PASS may still be a production FAIL if latency/retries/cost regress.
- **CI:** fast deterministic PR checks; expensive live LLM/MCP/browser checks on manual/nightly/release jobs.
