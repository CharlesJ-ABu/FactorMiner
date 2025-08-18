"""
核心层：评估结果读写IO
WebUI 仅调用此处API，不包含任何算法
"""

from pathlib import Path
from datetime import datetime
from typing import Dict, Any, Optional
import json


def _factorlib_dir() -> Path:
    return Path(__file__).parent.parent.parent / "factorlib"


def _evaluation_file(factor_id: str) -> Path:
    return _factorlib_dir() / "evaluations" / f"{factor_id}_evaluation.json"


def save_evaluation_results(factor_id: str, results: Dict[str, Any], metadata: Dict[str, Any]) -> None:
    """
    保存评估结果到 factorlib/evaluations/<factor_id>_evaluation.json
    结构为多结果列表：{"factor_id": ..., "evaluations": [{metadata, results, evaluated_at}, ...]}
    """
    eval_dir = _factorlib_dir() / "evaluations"
    eval_dir.mkdir(parents=True, exist_ok=True)

    evaluation_file = _evaluation_file(factor_id)

    new_entry = {
        'metadata': metadata or {},
        'results': results or {},
        'evaluated_at': datetime.now().isoformat()
    }

    payload: Optional[Dict[str, Any]] = None

    if evaluation_file.exists():
        try:
            with open(evaluation_file, 'r', encoding='utf-8') as f:
                existing = json.load(f)
        except Exception:
            existing = {}

        if 'evaluations' not in existing:
            upgraded = {
                'factor_id': existing.get('factor_id', factor_id),
                'evaluations': []
            }
            if 'results' in existing or 'metadata' in existing:
                upgraded['evaluations'].append({
                    'metadata': existing.get('metadata', {}),
                    'results': existing.get('results'),
                    'evaluated_at': existing.get('evaluated_at', datetime.now().isoformat())
                })
            existing = upgraded

        existing.setdefault('factor_id', factor_id)
        existing.setdefault('evaluations', []).append(new_entry)
        payload = existing
    else:
        payload = {
            'factor_id': factor_id,
            'evaluations': [new_entry]
        }

    with open(evaluation_file, 'w', encoding='utf-8') as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)


def load_evaluations(factor_id: str) -> Dict[str, Any]:
    """
    加载评估结果，统一返回 {factor_id, evaluations: [...]} 结构
    """
    evaluation_file = _evaluation_file(factor_id)
    if not evaluation_file.exists():
        return {'factor_id': factor_id, 'evaluations': []}

    try:
        with open(evaluation_file, 'r', encoding='utf-8') as f:
            payload = json.load(f)
    except Exception:
        return {'factor_id': factor_id, 'evaluations': []}

    if 'evaluations' in payload and isinstance(payload['evaluations'], list):
        return payload

    # 旧结构升级
    upgraded = {
        'factor_id': payload.get('factor_id', factor_id),
        'evaluations': []
    }
    if 'results' in payload or 'metadata' in payload:
        upgraded['evaluations'].append({
            'metadata': payload.get('metadata', {}),
            'results': payload.get('results'),
            'evaluated_at': payload.get('evaluated_at')
        })
    return upgraded


