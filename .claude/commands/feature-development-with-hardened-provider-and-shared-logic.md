---
name: feature-development-with-hardened-provider-and-shared-logic
description: Workflow command scaffold for feature-development-with-hardened-provider-and-shared-logic in ashare-llm-analyst.
allowed_tools: ["Bash", "Read", "Write", "Grep", "Glob"]
---

# /feature-development-with-hardened-provider-and-shared-logic

Use this workflow when working on **feature-development-with-hardened-provider-and-shared-logic** in `ashare-llm-analyst`.

## Goal

Implements a new or hardened data provider with shared logic, including core refactoring, provider implementation, and comprehensive tests.

## Common Files

- `src/core/anti_crawler.py`
- `src/data/downloaders.py`
- `src/data/providers.py`
- `src/core/cache.py`
- `src/core/exceptions.py`
- `tests/test_providers_harden.py`

## Suggested Sequence

1. Understand the current state and failure mode before editing.
2. Make the smallest coherent change that satisfies the workflow goal.
3. Run the most relevant verification for touched files.
4. Summarize what changed and what still needs review.

## Typical Commit Signals

- Refactor core logic (e.g., anti-crawler, retry manager) into a reusable module (e.g., src/core/anti_crawler.py)
- Update provider implementation to use shared logic and add new features (e.g., rate limiting, retry, caching, shared login) in src/data/providers.py
- Update or extend core utilities as needed (e.g., src/core/cache.py, src/core/exceptions.py)
- Write or update comprehensive tests for the provider, including offline and edge case coverage (e.g., tests/test_providers_harden.py, tests/test_providers.py)
- Register new or updated tests in the test runner (e.g., tests/run_tests.py)

## Notes

- Treat this as a scaffold, not a hard-coded script.
- Update the command if the workflow evolves materially.