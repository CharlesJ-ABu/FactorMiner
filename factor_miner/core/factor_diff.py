"""
核心层：因子挖掘结果与现有因子库对比
对比 definitions、functions、models，供上层决定是否保存
"""

from __future__ import annotations

import json
import hashlib
import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Any, List, Optional

import pandas as pd


def _factorlib_dir() -> Path:
    return Path(__file__).parent.parent.parent / "factorlib"


def _hash_obj(obj: Any) -> str:
    try:
        text = json.dumps(obj, sort_keys=True, ensure_ascii=False, default=str)
    except Exception:
        text = str(obj)
    return hashlib.md5(text.encode('utf-8')).hexdigest()[:12]


def _load_definition(factor_id: str) -> Optional[Dict[str, Any]]:
    f = _factorlib_dir() / "definitions" / f"{factor_id}.json"
    if not f.exists():
        return None
    try:
        return json.loads(f.read_text(encoding='utf-8'))
    except Exception:
        return None


def _load_function_code(definition: Dict[str, Any]) -> Optional[str]:
    comp = (definition or {}).get('computation_data') or {}
    rel = comp.get('function_file')
    if not rel:
        return None
    path = _factorlib_dir() / rel
    if not path.exists():
        return None
    try:
        return path.read_text(encoding='utf-8')
    except Exception:
        return None


def _load_model_meta(factor_id: str) -> Optional[Dict[str, Any]]:
    model_path = _factorlib_dir() / "models" / f"{factor_id}.pkl"
    if not model_path.exists():
        return None
    try:
        with open(model_path, 'rb') as f:
            art = pickle.load(f)
        model = art.get('model')
        params = None
        try:
            params = model.get_params() if hasattr(model, 'get_params') else None
        except Exception:
            params = None
        meta = {
            'model_class': model.__class__.__name__ if model is not None else None,
            'has_scaler': art.get('scaler') is not None,
            'feature_columns': art.get('feature_columns') or [],
            'params': params
        }
        meta['signature'] = _hash_obj({k: meta[k] for k in ['model_class', 'has_scaler', 'feature_columns', 'params']})
        return meta
    except Exception:
        return None


@dataclass
class DiffItem:
    factor_id: str
    status: str  # 'new'|'identical'|'different'|'missing_artifact'
    existing: Dict[str, Any]
    new: Dict[str, Any]


def compare_mined_factors_with_library(factors_df: pd.DataFrame) -> Dict[str, Any]:
    """
    对比本次挖掘生成的因子与因子库中的 definitions/functions/models
    返回用于决策保存的diff报告
    """
    items: List[Dict[str, Any]] = []

    for factor_id in list(factors_df.columns):
        existing_def = _load_definition(factor_id) or {}
        existing_comp_type = existing_def.get('computation_type')
        existing_func_code = _load_function_code(existing_def)
        existing_func_sig = _hash_obj(existing_func_code) if existing_func_code else None
        existing_model_meta = _load_model_meta(factor_id)

        # 预期新定义（对于挖掘产出默认为 ML 模型）
        candidate_model_meta = _load_model_meta(factor_id)
        new_def = {
            'factor_id': factor_id,
            'computation_type': 'ml_model' if candidate_model_meta else 'function',
            'computation_data': {
                'artifact_path': f"models/{factor_id}.pkl" if candidate_model_meta else None
            }
        }

        # 判定状态
        if not existing_def:
            status = 'new' if candidate_model_meta else 'missing_artifact'
        else:
            if new_def['computation_type'] == 'ml_model' and existing_comp_type == 'ml_model':
                # 比较模型签名
                new_sig = (candidate_model_meta or {}).get('signature')
                old_sig = (existing_model_meta or {}).get('signature')
                status = 'identical' if new_sig and old_sig and new_sig == old_sig else 'different'
            elif new_def['computation_type'] == 'function' and existing_comp_type == 'function':
                status = 'identical' if existing_func_sig else 'different'
            else:
                status = 'different'

        items.append({
            'factor_id': factor_id,
            'status': status,
            'existing': {
                'definition': existing_def,
                'model_meta': existing_model_meta,
                'function_signature': existing_func_sig
            },
            'new': {
                'definition': new_def,
                'model_meta': candidate_model_meta
            }
        })

    summary = {
        'total_mined': len(factors_df.columns),
        'new': sum(1 for it in items if it['status'] == 'new'),
        'identical': sum(1 for it in items if it['status'] == 'identical'),
        'different': sum(1 for it in items if it['status'] == 'different'),
        'missing_artifact': sum(1 for it in items if it['status'] == 'missing_artifact'),
    }

    return {'summary': summary, 'items': items}


