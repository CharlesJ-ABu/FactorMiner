import pandas as pd
from core.evaluation.targets import ForwardReturnTarget

def test_unknown_future_stays_missing_and_is_excluded_pairwise():
    frame=pd.DataFrame({"close":[100.,101.,102.,103.,104.,105.,106.,107.]})
    target=ForwardReturnTarget(horizon_bars=3).build(frame)
    factor=pd.Series(range(len(frame)),dtype=float)
    assert target.tail(3).isna().all()
    assert int((factor.notna() & target.notna()).sum()) == 5
    assert int((factor.notna() & target.fillna(0).notna()).sum()) == 8
