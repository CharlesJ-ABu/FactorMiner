import os
import tempfile
import unittest

import numpy as np
import pandas as pd

from core.evaluation.evaluator import ParallelEvaluator
from core.miner.entities import FactorMetadata
from core.execution.compiler import FactorCompiler
from core.miner.nn import NNDataAdapter, NumpyMLPFactorModel, load_nn_model
from core.storage.factor_storage import LocalFactorStorage
from user_workspace.custom_miners.my_custom_nn import MyCustomNNMiner
from user_workspace.custom_miners.my_custom_nn_temporal import (
    MyTemporalNNMiner,
    TemporalFeatureAdapter,
    TemporalICFactorModel,
)


class SplitDataClient:
    def __init__(self, train_data, train_returns, test_data, test_returns):
        self._train_data = train_data
        self._train_returns = train_returns
        self._test_data = test_data
        self._test_returns = test_returns

    def get_data(self):
        return self._train_data

    def get_returns(self):
        return self._train_returns

    def get_test_data(self):
        return self._test_data

    def get_test_returns(self):
        return self._test_returns


class NNMinerTests(unittest.TestCase):
    @staticmethod
    def _sequential_split():
        train_index = pd.date_range("2024-01-01", periods=120, freq="min")
        test_index = pd.date_range("2024-02-01", periods=80, freq="min")
        train_x = np.linspace(-2.0, 2.0, len(train_index))
        test_x = np.linspace(50.0, 54.0, len(test_index))
        train = pd.DataFrame(
            {
                "close": 100.0 + train_x,
                "volume": 1000.0 + 20.0 * np.sin(train_x),
            },
            index=train_index,
        )
        test = pd.DataFrame(
            {
                "close": 100.0 + test_x,
                "volume": 1000.0 + 20.0 * np.sin(test_x),
            },
            index=test_index,
        )
        train_returns = pd.Series(
            0.002 * train_x + 0.001 * np.sin(train_x * 3),
            index=train_index,
        )
        test_returns = pd.Series(
            0.002 * test_x + 0.001 * np.sin(test_x * 3),
            index=test_index,
        )
        return train, train_returns, test, test_returns

    @staticmethod
    def _cross_asset_split():
        def make_split(start, periods, offset):
            index = pd.date_range(start, periods=periods, freq="min")
            assets = pd.Index(["A", "B", "C"], name="asset")
            time_signal = np.linspace(-1.0, 1.0, periods)[:, None]
            asset_signal = np.asarray([0.1, 0.4, 0.8])[None, :]
            close = pd.DataFrame(
                100.0 + offset + time_signal + asset_signal,
                index=index,
                columns=assets,
            )
            volume = pd.DataFrame(
                1000.0 + 10.0 * time_signal - 5.0 * asset_signal,
                index=index,
                columns=assets,
            )
            returns = pd.DataFrame(
                0.003 * time_signal + 0.002 * asset_signal,
                index=index,
                columns=assets,
            )
            return {"close": close, "volume": volume}, returns

        train_data, train_returns = make_split("2024-01-01", 60, 0.0)
        test_data, test_returns = make_split("2024-02-01", 45, 5.0)
        return train_data, train_returns, test_data, test_returns

    def _mine(self, client, mode, iterations=2):
        config = {
            "hidden_dim": 4,
            "learning_rate": 0.02,
            "random_seed": 7,
            "nn_training_epochs": 20,
            "nn_min_samples": 30,
            "top_k_factors": 2,
            "max_corr": 0.9999,
            "data_feeds": {
                "required_streams": ["close", "volume"],
                "mining_mode": mode,
            },
        }
        miner = MyCustomNNMiner(client, config)
        miner.evaluator = ParallelEvaluator(client, config)
        return miner, miner.mine(iterations)

    def test_model_artifact_round_trip_reproduces_predictions(self):
        train, _, test, _ = self._sequential_split()
        model = NumpyMLPFactorModel(
            ["close", "volume"], hidden_dim=3, random_seed=11
        )
        model.fit_scaler(train[["close", "volume"]].to_numpy())
        before = model.predict_channel(test, 1)

        artifact = model.export_artifact()
        restored = NumpyMLPFactorModel.from_artifact(artifact.payload)
        restored_from_registry = load_nn_model(
            artifact.model_format, artifact.payload
        )
        after = restored.predict_channel(test, 1)
        registry_after = restored_from_registry.predict_channel(test, 1)

        self.assertEqual(artifact.model_format, "numpy_mlp_factor_v1")
        self.assertEqual(restored.features, ["close", "volume"])
        np.testing.assert_allclose(before.to_numpy(), after.to_numpy(), rtol=0, atol=0)
        np.testing.assert_allclose(
            before.to_numpy(), registry_after.to_numpy(), rtol=0, atol=0
        )

    def test_sequential_miner_uses_train_scaler_and_oos_metrics(self):
        split = self._sequential_split()
        client = SplitDataClient(*split)
        miner, factors = self._mine(client, "sequential_single")

        expected_mean = split[0][["close", "volume"]].to_numpy().mean(axis=0)
        np.testing.assert_allclose(miner.model.feature_mean, expected_mean)
        self.assertTrue(factors)
        self.assertTrue(all(factor.metrics["out_of_sample"] == 1.0 for factor in factors))
        self.assertTrue(all("train_IC" in factor.metrics for factor in factors))
        self.assertTrue(all(factor.model_version_id.startswith("nn_") for factor in factors))

    def test_cross_asset_adapter_and_miner_restore_dataframes(self):
        split = self._cross_asset_split()
        client = SplitDataClient(*split)
        miner, factors = self._mine(client, "cross_asset", iterations=1)

        self.assertTrue(factors)
        values = factors[0].compute(split[2])
        self.assertIsInstance(values, pd.DataFrame)
        self.assertEqual(values.shape, split[3].shape)
        self.assertEqual(list(values.columns), list(split[3].columns))
        self.assertEqual(miner.model.mode, "cross_asset")

    def test_adapter_rejects_missing_features(self):
        adapter = NNDataAdapter(["close", "volume"])
        data = pd.DataFrame(
            {"close": [1.0, 2.0]},
            index=pd.date_range("2024-01-01", periods=2, freq="min"),
        )
        with self.assertRaisesRegex(ValueError, "missing required features"):
            adapter.prepare(data)

    def test_miner_rejects_overlapping_train_and_test_splits(self):
        train, train_returns, _, _ = self._sequential_split()
        client = SplitDataClient(train, train_returns, train.copy(), train_returns.copy())
        with self.assertRaisesRegex(RuntimeError, "mine_period and test_period overlap"):
            self._mine(client, "sequential_single", iterations=1)

    def test_api_logic_supports_new_and_legacy_nn_artifacts(self):
        from api.main import _factor_logic

        with tempfile.TemporaryDirectory() as temp_dir:
            original_cwd = os.getcwd()
            try:
                os.chdir(temp_dir)
                storage = LocalFactorStorage()
                model = NumpyMLPFactorModel(["close", "volume"], 2)
                model.fit_scaler(np.asarray([[1.0, 2.0], [2.0, 4.0]]))
                artifact = model.export_artifact()
                filename = storage.save_model_artifact(
                    "nn_new", artifact.payload, artifact.extension
                )
                metadata = FactorMetadata("new", "MyCustomNN", "tester")
                storage.save_nn_factor_channel(
                    "nn_new",
                    0,
                    metadata,
                    artifact.logic_reference(filename),
                )
                new_logic = _factor_logic(storage.get_metadata("new"))
                self.assertTrue(new_logic["weights_available"])
                self.assertEqual(new_logic["model_format"], "numpy_mlp_factor_v1")
                compiled = FactorCompiler(storage).compile_for_live_trading("new")
                sample = pd.DataFrame(
                    {"close": [1.5], "volume": [3.0]},
                    index=pd.date_range("2024-01-01", periods=1),
                )
                self.assertIsInstance(compiled(sample), pd.Series)

                storage.save_model_weights("nn_legacy", b"legacy")
                legacy_metadata = FactorMetadata("legacy", "MyCustomNN", "tester")
                storage.save_nn_factor_channel("nn_legacy", 1, legacy_metadata)
                legacy_logic = _factor_logic(storage.get_metadata("legacy"))
                self.assertTrue(legacy_logic["weights_available"])
                self.assertEqual(legacy_logic["model_format"], "legacy_raw_weights")
            finally:
                os.chdir(original_cwd)

    def test_temporal_features_do_not_look_forward(self):
        train, _, _, _ = self._sequential_split()
        train = train.assign(
            open=train["close"] - 0.1,
            high=train["close"] + 0.2,
            low=train["close"] - 0.2,
        )
        cutoff = train.index[80]
        mutated = train.copy()
        mutated.loc[mutated.index > cutoff, "close"] *= 10.0
        adapter = TemporalFeatureAdapter()
        original_features = adapter.engineer_features(train)
        mutated_features = adapter.engineer_features(mutated)
        pd.testing.assert_series_equal(
            original_features.loc[cutoff],
            mutated_features.loc[cutoff],
            check_names=False,
        )

    def test_temporal_model_round_trip_and_five_minute_label(self):
        train, _, test, _ = self._sequential_split()
        for frame in (train, test):
            frame["open"] = frame["close"] - 0.1
            frame["high"] = frame["close"] + 0.2
            frame["low"] = frame["close"] - 0.2

        config = {
            "hidden_dim": 4,
            "learning_rate": 0.01,
            "random_seed": 9,
            "nn_training_epochs": 3,
            "target": {
                "type": "forward_return",
                "entry_price": "current_close",
                "exit_price": "close",
                "horizon_bars": 5,
                "return_type": "simple",
            },
            "nn_ic_loss_weight": 0.7,
            "data_feeds": {
                "required_streams": ["open", "high", "low", "close", "volume"],
                "mining_mode": "sequential_single",
            },
        }
        client = SplitDataClient(
            train,
            train["close"].pct_change().shift(-1),
            test,
            test["close"].pct_change().shift(-1),
        )
        miner = MyTemporalNNMiner(client, config)
        miner.initialize_search_space()
        target = miner.build_target_returns(train, client.get_returns(), "mine")
        expected = train["close"].shift(-5) / train["close"] - 1.0
        expected.name = "returns"
        pd.testing.assert_series_equal(target, expected)

        prepared = miner.model.adapter.prepare(train, target)
        miner.model.train(
            prepared.X,
            prepared.y,
            epochs=3,
            learning_rate=0.01,
        )
        before = miner.model.predict_channel(test, 0)
        artifact = miner.model.export_artifact()
        restored = TemporalICFactorModel.from_artifact(artifact.payload)
        after = restored.predict_channel(test, 0)
        np.testing.assert_allclose(
            before.to_numpy(), after.to_numpy(), rtol=0, atol=0, equal_nan=True
        )

    def test_temporal_fitness_preserves_ic_sign(self):
        from user_workspace.custom_fitness.my_temporal_positive_ic import (
            temporal_positive_ic_fitness,
        )

        values = pd.Series([1.0, 2.0, 3.0])
        self.assertLess(
            temporal_positive_ic_fitness(
                values, values, {"RankIC": -0.02}
            ),
            0.0,
        )


if __name__ == "__main__":
    unittest.main()
