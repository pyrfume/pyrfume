"""Unit tests for benchmarking.py."""
import unittest
import pandas as pd
from sklearn.metrics import SCORERS
from sklearn.preprocessing import MinMaxScaler
import pyrfume.benchmarking as pbm


class TestPyrfumeDataset(unittest.TestCase):
    """Unit tests for the PyrfumeDataset class."""

    def test_instance(self):
        cids = [4, 6, 11, 13, 19, 33, 34, 49, 51, 58, 66, 70, 72, 98]
        df = pd.DataFrame(
            data=["red"] * 6 + ["blue"] * 6 + ["green"] * 2, columns=["target"], index=cids
        ).rename_axis("CID")
        dataset = pbm.PyrfumeDataset(
            archive="test", df=df, task="classification", feature_set="mordred"
        )

        self.assertEqual(type(dataset).__name__, "PyrfumeDataset")
        self.assertEqual(dataset.archive, "test")
        self.assertEqual(dataset.feature_set, "mordred")
        self.assertEqual(dataset.n_splits, 2)
        self.assertEqual(dataset.task, "classification"),
        self.assertEqual(dataset.target_name, "target")
        self.assertEqual(dataset.label_encoder, None)
        self.assertTrue(
            all(
                col in dataset.get_feature_names()
                for col in ["ABC", "ABCGG", "nAcid", "nBase", "SpAbs_A"]
            )
        )
        self.assertEqual(dataset.get_cids(), cids)

        dataset.select_features(score_function="f-statistic", mode="k_best", param=5)
        self.assertEqual(dataset.df.shape[1] - 1, 5)

        dataset.threshold_labels(min_counts=6)
        self.assertNotIn("green", dataset.df["target"].values)
        self.assertEqual(dataset.n_splits, 3)

        dataset.set_n_splits(n_splits=5)
        self.assertEqual(dataset.n_splits, 5)

    def test_add_features(self):
        dataset = pbm.PyrfumeDataset(
            archive="test_data",
            df=pd.DataFrame({"target": [1, 2, 3, 4, 5]}).rename_axis("CID"),
            task="regression",
        )
        dataset.add_features(
            feature_set="mock_features",
            features=pd.DataFrame({"f1": [6, 7, 8, 9, 10], "f2": [11, 12, 13, 14, 15]}).rename_axis(
                "CID"
            ),
        )
        self.assertEqual(dataset.feature_set, "mock_features")
        self.assertEqual(dataset.df.columns.to_list(), ["target", "f1", "f2"])


class TestModel(unittest.TestCase):
    """Unit tests for the Model class."""

    def test_instance(self):
        model = pbm.Model("LinearRegression", param_grid={"fit_intercept": [True, False]})
        self.assertEqual(type(model.steps[0]).__name__, "LinearRegression")
        self.assertEqual(model.task, "regression")
        self.assertEqual(model.scoring, ["neg_root_mean_squared_error", "max_error", "r2"])
        self.assertIn("linearregression", model.param_grid)
        self.assertEqual(model.pipeline_string, "LinearRegression")

        model.add_feature_selection(
            score_function="f-statistic", mode="k_best", param_list=[10, 100]
        )
        self.assertEqual(model.pipeline_string.split(";")[0], "GenericUnivariateSelect")
        self.assertEqual(type(model.steps[0]).__name__, "GenericUnivariateSelect")
        self.assertIn("genericunivariateselect", model.param_grid)

        model.add_step(step=MinMaxScaler(), position=0)
        self.assertEqual(model.pipeline_string.split(";")[0], "MinMaxScaler")
        self.assertEqual(type(model.steps[0]).__name__, "MinMaxScaler")

        model.set_parameter_grid(param_grid={"minmaxscaler": {"clip": [True, False]}})
        self.assertIn("minmaxscaler", model.param_grid)
        self.assertNotIn("genericunivariateselect", model.param_grid)
        self.assertNotIn("linearregression", model.param_grid)


class TestBenchmarking(unittest.TestCase):
    """Unit tests for functions in pyrfume.benchmarking.py."""

    def setUp(self):
        self.mock_regression_data = pbm.PyrfumeDataset(
            archive="test_data",
            df=pd.DataFrame({"target": [1, 2, 3, 4, 5]}).rename_axis("CID"),
            task="regression",
        )
        self.mock_regression_data.add_features(
            feature_set="mock_features",
            features=pd.DataFrame({"f1": [6, 7, 8, 9, 10], "f2": [11, 12, 13, 14, 15]}).rename_axis(
                "CID"
            ),
        )

    def test_available_estimators(self):
        estimator_map = list(pbm.ESTIMATOR_MAP.keys())
        classifiers = list(pbm.DEFAULT_PARAMETERS_CLASSIFIERS.keys())
        regressors = list(pbm.DEFAULT_PARAMETERS_REGRESSORS.keys())
        self.assertListEqual(estimator_map, classifiers + regressors)

    def test_resolve_feature_selection(self):
        with self.assertRaises(KeyError):
            pbm.resolve_feature_selection(
                task="bad_task", score_function="f-statistic", mode="k_best"
            )
        with self.assertRaises(KeyError):
            pbm.resolve_feature_selection(
                task="classification", score_function="bad_score_func", mode="k_best"
            )
        selector = pbm.resolve_feature_selection(
            task="classification", score_function="f-statistic", mode="k_best"
        )
        self.assertEqual(type(selector).__name__, "GenericUnivariateSelect")
        self.assertEqual(selector.mode, "k_best")
        self.assertEqual(selector.score_func.__name__, "f_classif")

    def test_resolve_task(self):
        self.assertEqual(pbm.resolve_task("LogisticRegression"), "classification")
        self.assertEqual(pbm.resolve_task("Ridge"), "regression")
        with self.assertRaises(ValueError):
            pbm.resolve_task("BadEstimator")

    def test_get_default_parameters(self):
        self.assertIsInstance(pbm.get_default_parameters(), dict)
        for estimator in ["LinearRegression", "LogisticRegression"]:
            self.assertIn("fit_intercept", pbm.get_default_parameters(estimator))
        with self.assertRaises(ValueError):
            pbm.get_default_parameters("BadEstimator")

    def test_resolve_scoring(self):
        for task in ["classification", "regression"]:
            self.assertTrue(
                all(scorer in SCORERS for scorer in pbm.resolve_scoring(scoring=None, task=task))
            )
        self.assertEqual(pbm.resolve_scoring(scoring="f1_macro"), "f1_macro")
        with self.assertRaises(ValueError):
            pbm.resolve_scoring(scoring=None, task=None)
        with self.assertRaises(ValueError):
            pbm.resolve_scoring(scoring=None, task="bad_task")

    def test_pipeline_steps_to_string(self):
        self.assertTrue(pbm.Model("BernoulliNB").steps, "MinMaxScaler;BernoulliNB")

    def test_list_default_estimators(self):
        self.assertTrue(
            all(
                estimator in pbm.DEFAULT_PARAMETERS_CLASSIFIERS
                for estimator in pbm.list_default_estimators("classification")
            )
        )
        self.assertTrue(
            all(
                estimator in pbm.DEFAULT_PARAMETERS_REGRESSORS
                for estimator in pbm.list_default_estimators("regression")
            )
        )
        self.assertTrue(
            all(
                estimator
                in {**pbm.DEFAULT_PARAMETERS_CLASSIFIERS, **pbm.DEFAULT_PARAMETERS_REGRESSORS}
                for estimator in pbm.list_default_estimators()
            )
        )
        with self.assertRaises(ValueError):
            pbm.list_default_estimators("bad_task")

    def test_get_molecule_features(self):
        cids = pd.Index([326, 1049, 5318042])
        for feature_set, sample_cols in zip(
            ["mordred", "morgan"],
            [["nAromAtom", "nC", "MW"], ["CCCCC=COC(=O)CCCCCCCC", "C", "CC(C)(C)O"]],
        ):
            features = pbm.get_molecule_features(cids, feature_set)
            self.assertTrue(all(col in features.columns for col in sample_cols))
            # Check whether feature have been imputed
            self.assertEqual(features.isna().any().sum(), 0)

        with self.assertRaises(ValueError):
            pbm.get_molecule_features(cids, "bad_features")

    def test_resolve_cv(self):
        for task, name in zip(["regression", "classification"], ["KFold", "StratifiedKFold"]):
            self.assertEqual(type(pbm.resolve_cv(task=task)).__name__, name)
        self.assertEqual(pbm.resolve_cv(task="classification", n_splits=10).n_splits, 10)
        with self.assertRaises(ValueError):
            pbm.resolve_cv(task="bad_features")

    def test_reformat_parameter_grid(self):
        model = pbm.Model(
            estimator="BernoulliNB", param_grid={"alpha": [1, 5, 10], "fit_prior": [True, False]}
        )
        model.add_feature_selection(
            score_function="f-statistic", mode="k_best", param_list=[100, 200, 300]
        )
        self.assertDictEqual(
            pbm.reformat_param_grid(model.param_grid),
            {
                "bernoullinb__alpha": [1, 5, 10],
                "bernoullinb__fit_prior": [True, False],
                "genericunivariateselect__param": [100, 200, 300],
            },
        )

    def test_evaluate_model(self):
        gs = pbm.evaluate_model(
            dataset=self.mock_regression_data, pipeline=pbm.Model("LinearRegression")
        )
        self.assertEqual(type(gs).__name__, "GridSearchCV")

    def test_evaluate_dummy_model(self):
        self.mock_regression_data.set_n_splits(2)
        df = pbm.evaluate_dummy_model(self.mock_regression_data)
        self.assertIsInstance(df, pd.DataFrame)
        self.assertTrue(list(df.index) == ["mean", "median"])
        self.assertTrue(list(df.columns) == ["neg_root_mean_squared_error", "max_error", "r2"])

    def test_gridsearch_results_to_dataframe(self):
        pipeline = pbm.Model("LinearRegression")
        gs = pbm.evaluate_model(dataset=self.mock_regression_data, pipeline=pipeline)
        df = pbm.gridsearch_results_to_dataframe(gs)
        self.assertIsInstance(df, pd.DataFrame)
        self.assertTrue(
            all(col in df.columns for col in ["pipeline_steps", "pipeline_string", "param_string"])
        )
        self.assertEqual(df.iloc[0]["pipeline_steps"], pipeline.steps)
        self.assertTrue(
            all(
                (col.startswith("mean_") or col.startswith("std_"))
                for col in df.columns
                if col not in ["pipeline_steps", "pipeline_string", "param_string"]
            )
        )

    def test_remove_prefix(self):
        self.assertEqual(pbm.remove_prefix("prefix", "pre"), "fix")
        self.assertEqual(pbm.remove_prefix("prefix", "test"), "prefix")

    def test_get_best_results(self):
        self.mock_regression_data.set_n_splits(2)
        gs = pbm.evaluate_model(
            dataset=self.mock_regression_data, pipeline=pbm.Model("LinearRegression")
        )
        df = pbm.gridsearch_results_to_dataframe(gs)
        best = pbm.get_best_results(df, metric="max_error", include_pipeline_steps=True)
        self.assertIsInstance(best, pd.DataFrame)
        self.assertTrue(
            ["max_error", "max_error_param_string", "pipeline_steps"] == best.columns.to_list()
        )

    def test_verify_batch_settings(self):
        pass

    def test_batch_gridsearchsv(self):
        pass

    def test_reconstruct_pipeline(self):
        model = pbm.Model("LinearRegression")
        param_string = "linearregression__fit_intercept=True"
        pipeline = pbm.reconstruct_pipeline(pipeline_steps=model.steps, param_string=param_string)
        self.assertEqual(type(pipeline).__name__, "Pipeline")
        self.assertEqual(model.steps, list(pipeline.named_steps.values()))
        self.assertDictContainsSubset(
            dict(item.split("=") for item in param_string.split(";")), pipeline.get_params()
        )

    def test_reconstruct_model(self):
        pipeline = pbm.reconstruct_model(
            dataset=self.mock_regression_data,
            pipeline_steps=pbm.Model("LinearRegression").steps,
            param_string="linearregression__fit_intercept=True",
        )
        self.assertEqual(type(pipeline).__name__, "Pipeline")
        self.assertEqual(pipeline.n_features_in_, 2)

        results = pd.DataFrame(
            {
                "max_error": {"LinearRegression": 0.0},
                "max_error_param_string": {
                    "LinearRegression": "linearregression__fit_intercept=True"
                },
                "pipeline_steps": {"LinearRegression": pbm.Model("LinearRegression").steps},
            }
        )
        pipeline = pbm.reconstruct_model(
            dataset=self.mock_regression_data, results=results, metric="max_error"
        )
        self.assertEqual(type(pipeline).__name__, "Pipeline")
        self.assertEqual(pipeline.n_features_in_, 2)

        with self.assertRaises(ValueError):
            pbm.reconstruct_model(self.mock_regression_data)

    def test_apply_model(self):
        pipeline = pbm.reconstruct_model(
            dataset=self.mock_regression_data,
            pipeline_steps=pbm.Model("LinearRegression").steps,
            param_string="linearregression__fit_intercept=True",
        )
        df = pbm.apply_model(dataset=self.mock_regression_data, model=pipeline)
        self.assertIsInstance(df, pd.DataFrame)
        self.assertEqual(df.columns.to_list(), ["target", "prediction", "%_error"])


if __name__ == "__main__":
    unittest.main()
