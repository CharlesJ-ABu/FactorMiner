import json
import time
from pathlib import Path
import pandas as pd

from factor_miner.core.factor_engine import get_global_engine
from factor_miner.core.factor_storage import get_global_storage


def find_any_feather(data_root: Path) -> Path:
    if data_root.exists():
        files = sorted(data_root.rglob('*.feather'))
        if files:
            return files[0]
    return None


def main():
    project_root = Path(__file__).resolve().parents[1]
    factorlib_dir = project_root / 'factorlib'
    data_root = project_root / 'data'

    # 选择一个样本数据文件
    preferred = [
        data_root / 'binance' / 'futures' / 'BTC_USDT_USDT-1h-futures.feather',
        data_root / 'binance' / 'futures' / 'BTC_USDT_USDT-15m-futures.feather',
        data_root / 'binance' / 'spot' / 'BTC_USDT-1h-spot.feather',
    ]
    sample_file = None
    for p in preferred:
        if p.exists():
            sample_file = p
            break
    if sample_file is None:
        sample_file = find_any_feather(data_root)

    if sample_file is None:
        print('No sample feather data file found under data/. Abort health check.')
        return

    df = pd.read_feather(sample_file)
    # Try to set time index if column exists
    for col in ['timestamp', 'datetime', 'time', 'date']:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors='coerce', utc=True)
            df = df.set_index(col)
            break

    storage = get_global_storage()
    engine = get_global_engine()

    results = []
    for factor_id in storage.list_factors():
        start = time.time()
        status = 'ok'
        nnz = None
        length = None
        error = None
        try:
            series = engine.compute_single_factor(factor_id, df)
            if series is None:
                status = 'empty'
            else:
                length = int(series.shape[0])
                nnz = int(series.fillna(0).ne(0).sum())
        except Exception as e:
            status = 'error'
            error = str(e)
        elapsed = time.time() - start
        results.append({
            'factor_id': factor_id,
            'status': status,
            'length': length,
            'non_zero': nnz,
            'elapsed_ms': int(elapsed * 1000),
            'error': error,
        })

    exports_dir = factorlib_dir / 'exports'
    exports_dir.mkdir(parents=True, exist_ok=True)
    out_path = exports_dir / 'factorlib_health_report.json'
    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump({'sample_file': str(sample_file), 'results': results}, f, ensure_ascii=False, indent=2)

    ok = sum(1 for r in results if r['status'] == 'ok')
    err = sum(1 for r in results if r['status'] == 'error')
    empty = sum(1 for r in results if r['status'] == 'empty')
    print(f'Health check done. ok={ok}, error={err}, empty={empty}. Report: {out_path}')


if __name__ == '__main__':
    main()


