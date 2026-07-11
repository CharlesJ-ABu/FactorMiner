import os
import logging
import pandas as pd
from typing import Dict, List, Any
from pathlib import Path

logger = logging.getLogger(__name__)

class RealDataClient:
    def __init__(self, config: Dict):
        self.config = config
        self.data_feed_config = config.get("data_feeds", {})
        
        self.exchange = self.data_feed_config.get("exchange", "binance")
        self.instrument_type = self.data_feed_config.get("instrument_type", "futures")
        self.timeframe = self.data_feed_config.get("timeframe", "1m")
        self.pairs = self.data_feed_config.get("pairs", [])
        self.mine_periods = self.data_feed_config.get("mine_period", [])
        self.test_periods = self.data_feed_config.get("test_period", [])
        self.mining_mode = self.data_feed_config.get("mining_mode", "sequential_single")
        
        # 预先加载好的缓存
        self._mine_data = None
        self._test_data = None
        
        # 在初始化时一次性加载数据
        self._load_all_data()

    def _load_all_data(self):
        if not self.pairs:
            logger.warning("No pairs specified in config data_feeds.")
            return

        if self.mining_mode == "sequential_single":
            primary_pair = self.pairs[0]
            if len(self.pairs) > 1:
                logger.info(f"Mining mode is 'sequential_single'. Only using the first pair: {primary_pair}")
            
            self._mine_data = self._load_periods_for_pair(primary_pair, self.mine_periods)
            self._test_data = self._load_periods_for_pair(primary_pair, self.test_periods)
            
        elif self.mining_mode == "cross_asset":
            logger.info("Mining mode is 'cross_asset'. Loading and pivoting data for all pairs.")
            
            def load_and_pivot(periods):
                dfs = []
                for pair in self.pairs:
                    df = self._load_periods_for_pair(pair, periods)
                    if not df.empty:
                        df['asset'] = pair
                        dfs.append(df)
                        
                if not dfs:
                    return None
                    
                combined = pd.concat(dfs)
                
                # Pivot into a dict of DataFrames (rows: date, cols: asset)
                pivot_dict = {}
                features = [c for c in combined.columns if c not in ['asset', 'date', 'time', 'timestamp']]
                for feature in features:
                    pivot_dict[feature] = combined.pivot(columns='asset', values=feature)
                    
                return pivot_dict
                
            self._mine_data = load_and_pivot(self.mine_periods)
            self._test_data = load_and_pivot(self.test_periods)
            
        else:
            logger.warning(f"Mining mode '{self.mining_mode}' is not fully supported yet. Falling back to sequential_single.")
            primary_pair = self.pairs[0]
            self._mine_data = self._load_periods_for_pair(primary_pair, self.mine_periods)
            self._test_data = self._load_periods_for_pair(primary_pair, self.test_periods)

    def _load_periods_for_pair(self, pair: str, periods: List[List[str]]) -> pd.DataFrame:
        if not periods:
            return pd.DataFrame()
            
        # 严格按照 batch_downloader.py 的本地数据存储命名进行匹配
        clean_symbol = pair.replace('/', '_').replace(':', '_')
        filename = f"{clean_symbol}-{self.timeframe}-{self.instrument_type}.feather"
        
        project_root = Path(os.getcwd())
        file_path = project_root / "data" / self.exchange / self.instrument_type / filename
        
        if not file_path.exists():
            logger.warning(f"Data file not found: {file_path}. Attempting to download automatically...")
            
            # 收集所有需要的数据段以计算最大的时间范围
            all_periods = self.mine_periods + self.test_periods
            if not all_periods:
                logger.error(f"Cannot download {pair}: no periods specified.")
                return pd.DataFrame()
                
            all_dates = []
            for p in all_periods:
                if len(p) >= 2:
                    all_dates.append(pd.to_datetime(p[0]))
                    all_dates.append(pd.to_datetime(p[1]))
                    
            if not all_dates:
                return pd.DataFrame()
                
            start_date = min(all_dates).strftime('%Y-%m-%d')
            end_date = max(all_dates).strftime('%Y-%m-%d')
            
            from core.data_feed.data_downloader import DataDownloader
            downloader = DataDownloader()
            
            # Binance symbol format uses normal pairing e.g. BTC/USDT
            download_symbol = pair.replace('_', '/')
            if not download_symbol.endswith('/USDT'):
                download_symbol += '/USDT'
                
            res = downloader.download_ohlcv(
                symbol=download_symbol,
                timeframe=self.timeframe,
                start_date=start_date,
                end_date=end_date,
                trade_type=self.instrument_type
            )
            
            if not file_path.exists():
                logger.error(f"Failed to auto-download {pair} data or file still missing.")
                return pd.DataFrame()
            else:
                logger.info(f"Successfully auto-downloaded data for {pair}!")
            
        logger.info(f"Loading data from {file_path}")
        df = pd.read_feather(file_path)
        
        # 标准化索引
        if 'date' in df.columns:
            df.set_index('date', inplace=True)
        elif 'time' in df.columns:
            df.set_index('time', inplace=True)
        elif 'timestamp' in df.columns:
            df['date'] = pd.to_datetime(df['timestamp'], unit='ms')
            df.set_index('date', inplace=True)
            df.drop('timestamp', axis=1, inplace=True)
            
        if not isinstance(df.index, pd.DatetimeIndex):
            df.index = pd.to_datetime(df.index)
            
        # 根据时间段进行过滤并拼接
        sliced_dfs = []
        for period in periods:
            if len(period) == 2:
                start_dt, end_dt = pd.to_datetime(period[0]), pd.to_datetime(period[1])
                sliced_df = df[(df.index >= start_dt) & (df.index <= end_dt)]
                sliced_dfs.append(sliced_df)
                
        if sliced_dfs:
            final_df = pd.concat(sliced_dfs)
            # 添加 Forward Return（下一根 K 线的收益率）作为因子评估标签
            # ⚠️ 必须使用 shift(-1)：因子只能用当期的 OHLCV，而 returns 必须是「下一期」的涨跌幅
            # 若使用同期 pct_change()，因子中的 close_t 与 returns_t 天然相关，导致 IC 虚高（数据穿越）
            if 'close' in final_df.columns:
                final_df['returns'] = final_df['close'].pct_change().shift(-1).fillna(0)
            return final_df
            
        return pd.DataFrame()

    def get_data(self) -> Any:
        if self._mine_data is None:
            logger.warning("Mine data is empty!")
            
        # Check if DataFrame (sequential) or empty DataFrame
        if isinstance(self._mine_data, pd.DataFrame) and self._mine_data.empty:
            logger.warning("Mine data is empty DataFrame!")
            
        return self._mine_data

    def get_returns(self) -> Any:
        if self.mining_mode == "cross_asset":
            return self._mine_data.get('returns') if self._mine_data else pd.DataFrame()
            
        if self._mine_data is not None and 'returns' in self._mine_data.columns:
            return self._mine_data['returns']
        return pd.Series(dtype=float)

    def get_test_data(self) -> Any:
        return self._test_data

    def get_test_returns(self) -> Any:
        if self.mining_mode == "cross_asset":
            return self._test_data.get('returns') if self._test_data else pd.DataFrame()
            
        if self._test_data is not None and 'returns' in self._test_data.columns:
            return self._test_data['returns']
        return pd.Series(dtype=float)
