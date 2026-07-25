"""Optional, append-only experiment recording for custom LLM miners."""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List

import numpy as np
import pandas as pd

from core.miner.expressions import FactorExpressionCode


class LLMExperimentRecorder:
    """Keep research evidence out of the user-facing miner implementation."""

    def __init__(
        self,
        config: Dict[str, Any],
        population_size: int,
        resolved_api_config: Dict[str, Any] | None = None,
        paradigm_name: str = "MyCustomLLM",
    ) -> None:
        self.config = config
        self.population_size = population_size
        self.event_index = 0
        self.generated_by_epoch: Dict[int, List[FactorExpressionCode]] = {}

        experiment = config.get("experiment", {})
        record_dir = experiment.get("record_dir")
        self.directory = Path(record_dir).resolve() if record_dir else None
        run_started_at = datetime.now(timezone.utc)
        self.run_id = run_started_at.strftime("run_%Y%m%dT%H%M%S_%fZ")
        if self.directory and experiment.get("create_run_subdir", False):
            self.directory = self.directory / self.run_id
        if not self.directory:
            return

        self.directory.mkdir(parents=True, exist_ok=True)
        llm_config = config.get("llm_api_config", {})
        manifest = {
            "schema_version": 1,
            "run_id": self.run_id,
            "experiment_name": experiment.get("name", "llm_factor_mining"),
            "started_at": run_started_at.isoformat(),
            "paradigm": config.get("paradigm", paradigm_name),
            "population_size": population_size,
            "max_iterations": config.get("max_iterations"),
            "top_k_factors": config.get("top_k_factors", population_size),
            "data_feeds": config.get("data_feeds", {}),
            "evaluation": config.get("evaluation", {}),
            "llm": {
                "base_url": (resolved_api_config or {}).get(
                    "base_url", llm_config.get("base_url", "")
                ),
                "model": (resolved_api_config or {}).get(
                    "model", llm_config.get("model", "")
                ),
                "temperature": 0.7,
                "require_live_api": bool(experiment.get("require_live_api", False)),
                "allow_fallback": bool(experiment.get("allow_fallback", True)),
                "key_environment_variables": llm_config.get("keys_env", []),
                "model_environment_variable": llm_config.get("model_env", ""),
                "base_url_environment_variable": llm_config.get(
                    "base_url_env", ""
                ),
            },
        }
        (self.directory / "manifest.json").write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2, default=str),
            encoding="utf-8",
        )

    @property
    def enabled(self) -> bool:
        return self.directory is not None

    @staticmethod
    def _json_default(value: Any) -> Any:
        if isinstance(value, np.integer):
            return int(value)
        if isinstance(value, np.floating):
            return float(value)
        if isinstance(value, (pd.Timestamp, datetime)):
            return value.isoformat()
        return str(value)

    def record(self, event_type: str, **payload: Any) -> None:
        if not self.directory:
            return
        self.event_index += 1
        event = {
            "event_index": self.event_index,
            "event_type": event_type,
            "recorded_at": datetime.now(timezone.utc).isoformat(),
            **payload,
        }
        with (self.directory / "events.jsonl").open(
            "a", encoding="utf-8"
        ) as handle:
            handle.write(
                json.dumps(
                    event,
                    ensure_ascii=False,
                    default=self._json_default,
                )
                + "\n"
            )

    def generation_started(
        self,
        epoch: int,
        prompt_hash: str,
        prompt: str,
        reflection_before: str,
    ) -> None:
        self.record(
            "generation_started",
            epoch=epoch,
            prompt_sha256=prompt_hash,
            prompt=prompt,
            reflection_before=reflection_before,
        )

    def generation_failed(
        self, epoch: int, candidate_index: int, reason: str
    ) -> None:
        self.record(
            "generation_failed",
            epoch=epoch,
            candidate_index=candidate_index,
            reason=reason,
        )

    def candidate_generated(
        self,
        expression: FactorExpressionCode,
        epoch: int,
        candidate_index: int,
        raw_response: str | None,
        generation_source: str,
    ) -> None:
        provenance = expression.get_provenance()
        self.record(
            "candidate_generated",
            epoch=epoch,
            candidate_index=candidate_index,
            candidate_id=provenance.get("candidate_id"),
            generation_source=generation_source,
            raw_response=raw_response,
            extracted_code=expression.get_source(),
            prompt_sha256=provenance.get("prompt_sha256"),
        )

    def generation_completed(
        self, epoch: int, candidates: Iterable[FactorExpressionCode]
    ) -> None:
        remembered = list(candidates)
        self.generated_by_epoch[epoch] = remembered
        self.record(
            "generation_completed",
            epoch=epoch,
            candidate_count=len(remembered),
        )

    @staticmethod
    def candidate_diagnostics(
        expression: FactorExpressionCode, data_source: Any
    ) -> Dict[str, Any]:
        data = (
            data_source.get_data()
            if hasattr(data_source, "get_data")
            else data_source
        )
        first = expression.compute(data)
        second = expression.compute(data)
        finite = np.isfinite(first.to_numpy(dtype=float, na_value=np.nan))
        output_shape = (
            [len(first)] if isinstance(first, pd.Series) else list(first.shape)
        )
        digest_payload = pd.util.hash_pandas_object(
            first, index=True
        ).to_numpy().tobytes()
        code = expression.get_source()
        return {
            "coverage": float(finite.sum() / finite.size) if finite.size else 0.0,
            "output_shape": output_shape,
            "deterministic_replay_equal": bool(first.equals(second)),
            "output_sha256": hashlib.sha256(digest_payload).hexdigest(),
            "lines_of_code": len(
                [line for line in code.splitlines() if line.strip()]
            ),
            "code_characters": len(code),
        }

    def candidate_succeeded(
        self,
        expression: FactorExpressionCode,
        epoch: int,
        metrics: Dict[str, Any],
        data_source: Any,
    ) -> None:
        try:
            diagnostics = self.candidate_diagnostics(expression, data_source)
        except Exception as exc:
            diagnostics = {
                "diagnostic_error_type": type(exc).__name__,
                "diagnostic_error": str(exc),
            }
        self.record(
            "candidate_evaluated",
            epoch=epoch,
            candidate_id=expression.get_provenance().get("candidate_id"),
            status="success",
            code=expression.get_source(),
            metrics=metrics,
            diagnostics=diagnostics,
        )

    def candidate_failed(
        self,
        expression: FactorExpressionCode,
        epoch: int,
        error_type: str,
        error_message: str,
    ) -> None:
        self.record(
            "candidate_evaluated",
            epoch=epoch,
            candidate_id=expression.get_provenance().get("candidate_id"),
            status="error",
            code=expression.get_source(),
            error_type=error_type,
            error_message=error_message,
        )

    def reflection_updated(
        self,
        epoch: int,
        successful_count: int,
        failed_count: int,
        reflection_after: str,
    ) -> None:
        self.record(
            "reflection_updated",
            epoch=epoch,
            successful_count=successful_count,
            failed_count=failed_count,
            reflection_after=reflection_after,
        )

    def archive_updated(
        self, epoch: int, candidates: Iterable[FactorExpressionCode]
    ) -> None:
        self.record(
            "archive_updated",
            epoch=epoch,
            retained_candidates=[
                {
                    "candidate_id": candidate.get_provenance().get("candidate_id"),
                    "fitness_score": candidate.metrics.get("fitness_score"),
                }
                for candidate in candidates
            ],
        )

    def epoch_completed(
        self, epoch: int, evaluated: Iterable[FactorExpressionCode]
    ) -> None:
        evaluated_candidates = list(evaluated)
        evaluated_ids = {
            candidate.get_provenance().get("candidate_id")
            for candidate in evaluated_candidates
        }
        generated = self.generated_by_epoch.get(epoch, [])
        for candidate in generated:
            candidate_id = candidate.get_provenance().get("candidate_id")
            if candidate_id not in evaluated_ids:
                self.record(
                    "candidate_filtered",
                    epoch=epoch,
                    candidate_id=candidate_id,
                    code=candidate.get_source(),
                    reason="logic_hash_deduplication",
                )
        self.record(
            "epoch_completed",
            epoch=epoch,
            generated_count=len(generated),
            evaluated_count=len(evaluated_candidates),
        )

    def run_completed(
        self,
        completed_iterations: int,
        candidates: Iterable[FactorExpressionCode],
    ) -> None:
        self.record(
            "run_completed",
            completed_iterations=completed_iterations,
            best_candidates=[
                {
                    "candidate_id": candidate.get_provenance().get("candidate_id"),
                    "metrics": candidate.metrics,
                    "code": candidate.get_source(),
                }
                for candidate in candidates
            ],
        )
