# LLM Factor Guardrails Review — 2026-07-24

## Scope

This review covers LLM-generated factor execution, candidate/result association,
reflection feedback, provenance persistence, and compatibility with GP, RL, NN,
Inspector, API, and Web UI contracts.

## Review passes

1. **Execution security**
   - Removed the unsandboxed `FactorExpressionCode` fallback.
   - Replaced substring blocking with an AST allowlist.
   - Confirmed imports, file access, dunder access, loops, invalid outputs, and
     missing `factor` assignments are rejected.

2. **Data correctness**
   - Enforced exact factor type, shape, index/columns, numeric dtype, and finite
     output requirements.
   - Confirmed `df["open"]` remains legal and index realignment is rejected.

3. **Evaluation association**
   - Bound every outcome to its originating candidate with
     `CandidateEvaluation`.
   - Updated GP and LLM consumers while preserving the existing feedback views
     used by RL, NN, and other callers.

4. **Isolation, lifecycle, and provenance**
   - Executed generated code in short-lived spawned processes with wall-clock,
     CPU, memory, and file-descriptor limits.
   - Added process termination coverage, bounded reflection memory, failed-case
     feedback, prompt hashes, and recursive credential removal before storage.

5. **Compatibility and integration**
   - Restored stored and ad-hoc LLM factors through the same sandbox in
     Inspector.
   - Corrected Inspector's LLM source directory from `src` to `sources`.
   - Verified API and Inspector imports and ran the complete Python test suite.

## Verification

- `python -m compileall -q core user_workspace tests`
- `python -m pytest -q tests`
  - Result: `41 passed, 8 subtests passed`
- `git diff --check`
- API and Inspector import smoke test passed.

## Follow-up cleanup — 2026-07-25

- Removed API-key fragments from rate-limit logs and redact bounded API error
  diagnostics.
- Repaired the TypeScript project references and source type errors; frontend
  lint and production build now complete successfully.
- Added `load_llm_source()` to the storage contract and removed Inspector's
  dependency on `LocalFactorStorage` internals.
- Reconstructed persisted GP and LLM artifacts as callable factors; live
  deployment now fails explicitly until a real transport exists.
- Made evaluator concurrency configurable through `evaluation.max_workers`
  with a bounded range of 1–64.
- Added strict cross-asset LLM DataFrame input/output contracts.
- Updated the architecture and user documentation to remove the obsolete
  string-blacklist sandbox example.
- Deprecated the legacy `DL` public name in favor of the canonical `NN`
  paradigm; new model-channel metadata uses `nn_channel` while historical
  `dl_channel` records remain readable.

Follow-up verification:

- `python -m pytest -q tests`
  - Result: `51 passed, 8 subtests passed`
- `npm run lint`
- `npm run build`
