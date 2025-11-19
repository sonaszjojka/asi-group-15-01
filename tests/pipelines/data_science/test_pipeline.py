"""
This is a boilerplate test file for pipeline 'data_science'
generated using Kedro 1.0.0.
Please add your pipeline tests here.

Kedro recommends using `pytest` framework, more info about it can be found
in the official documentation:
https://docs.pytest.org/en/latest/getting-started.html
"""

import pandas as pd
import numpy as np
import pytest
from pathlib import Path
from unittest.mock import MagicMock, patch

from src.asi_group_15_01.pipelines.data_science.nodes import (
    basic_clean,
    train_test_split,
    evaluate_autogluon,
)

NODES_PATH = "src.asi_group_15_01.pipelines.data_science.nodes"


# --- Fixtures for testing ---


@pytest.fixture
def sample_raw_data():
    """
    Creates a sample raw DataFrame to simulate input data for testing.
    """

    data = {
        "age": [39, 50, 38, 50, 40, 39, 39],
        "workclass": [
            "State-gov",
            "Self-emp-not-inc",
            "Private",
            "Private",
            "?",
            "Private",
            "Private",
        ],
        "fnlwgt": [77516, 83311, 215646, 215646, 120000, 120000, 120000],
        "education": [
            "Bachelors",
            "Bachelors",
            "HS-grad",
            "HS-grad",
            "Bachelors",
            "Bachelors",
            "Bachelors",
        ],
        "marital-status": [
            "Never-married",
            "Married-civ-spouse",
            "Divorced",
            "Married-civ-spouse",
            "Married-civ-spouse",
            "Married-civ-spouse",
            "Married-civ-spouse",
        ],
        "occupation": [
            "Adm-clerical",
            "Exec-managerial",
            "Handlers-cleaners",
            "Handlers-cleaners",
            "Exec-managerial",
            "Exec-managerial",
            "Exec-managerial",
        ],
        "relationship": [
            "Not-in-family",
            "Husband",
            "Not-in-family",
            "Husband",
            "Husband",
            "Husband",
            "Husband",
        ],
        "race": ["White", "White", "White", "White", "White", "White", "White"],
        "sex": ["Male", "Male", "Male", "Male", "Male", "Male", "Male"],
        "capital-gain": [2174, 0, 0, 0, 0, 0, 0],
        "capital-loss": [0, 0, 0, 0, 1000, 0, 0],
        "hours-per-week": [40, 13, 40, 40, 40, 40, 40],
        "native-country": [
            "United-States",
            "United-States",
            "United-States",
            "United-States",
            "United-States",
            "United-States",
            "United-States",
        ],
        "income": ["<=50K", "<=50K", "<=50K", "<=50K", ">50K", ">50K", ">50K"],
    }
    df = pd.DataFrame(data)
    return df


@pytest.fixture
def split_params():
    """
    Sample split parameters for train_test_split function.
    """
    return {"test_size": 0.3, "random_state": 42, "stratify": False}


class TestDataSciencePipeline:

    # Tests for node: basic_clean

    def test_basic_clean_removes_nans_and_question_marks(self, sample_raw_data):
        """
        Tests that basic_clean removes rows with '?'.
        """

        cleaned_df = basic_clean(sample_raw_data.copy())

        assert (
            len(cleaned_df) == 5
        ), "Number of rows after cleaning (duplicates and NaN) is incorrect."
        assert (
            not cleaned_df.isnull().any().any()
        ), "There are still NaN values in the dataframe."

    def test_basic_clean_drops_unnecessary_columns(self, sample_raw_data):
        """
        Tests that basic_clean drops 'fnlwgt' and 'education' columns.
        """

        cleaned_df = basic_clean(sample_raw_data.copy())

        assert "fnlwgt" not in cleaned_df.columns, "Column 'fnlwgt' was not removed."
        assert (
            "education" not in cleaned_df.columns
        ), "Column 'education' was not removed."

    def test_basic_clean_performs_target_encoding(self, sample_raw_data):
        """
        Tests that the target column is encoded and the original is removed.
        """

        cleaned_df = basic_clean(sample_raw_data.copy())

        assert (
            "income_encoded" in cleaned_df.columns
        ), "Column 'income_encoded' is missing."
        assert (
            "income" not in cleaned_df.columns
        ), "Original target column 'income' was not removed."
        assert (
            cleaned_df["income_encoded"].dtype == np.int64
            or cleaned_df["income_encoded"].dtype == np.uint8
        ), "Column 'income_encoded' should be numeric."

    def test_basic_clean_performs_log_transformation(self, sample_raw_data):
        """
        Tests that 'capital-gain' and 'capital-loss' columns are log-transformed and originals are removed.
        """
        cleaned_df = basic_clean(sample_raw_data.copy())

        assert (
            "capital-gain-log" in cleaned_df.columns
        ), "Column 'capital-gain-log' is missing."
        assert (
            "capital-loss-log" in cleaned_df.columns
        ), "Column 'capital-loss-log' is missing."
        assert (
            "capital-gain" not in cleaned_df.columns
        ), "Original column 'capital-gain' was not removed."
        assert (
            "capital-loss" not in cleaned_df.columns
        ), "Original column 'capital-loss' was not removed."

    # Tests for node: train_test_split

    def test_train_test_split_sizes(self, sample_raw_data, split_params):
        """
        Tests that train/test split maintains the proportions from 'test_size'.
        """

        df_clean = basic_clean(sample_raw_data.copy())

        X_train, X_test, y_train, y_test = train_test_split(df_clean, split_params)

        assert len(X_train) == 3, "Size of X_train is incorrect."
        assert len(X_test) == 2, "Size of X_test is incorrect."
        assert len(y_train) == 3, "Size of y_train is incorrect."
        assert len(y_test) == 2, "Size of y_test is incorrect."

    def test_train_test_split_no_target_leakage(self, sample_raw_data, split_params):
        """
        Tests that the target column has been removed from feature sets (X_train, X_test).
        """

        df_clean = basic_clean(sample_raw_data.copy())

        X_train, X_test, y_train, y_test = train_test_split(df_clean, split_params)

        assert (
            "income_encoded" not in X_train.columns
        ), "Target leakage occurred in X_train."
        assert (
            "income_encoded" not in X_test.columns
        ), "Target leakage occurred in X_test."
        assert (
            "income_encoded" in y_train.columns
        ), "y_train does not contain the target column."
        assert (
            "income_encoded" in y_test.columns
        ), "y_test does not contain the target column."


@pytest.fixture
def sample_raw_data_auto_gluon():
    """
    Creates a sample DataFrames for testing.
    """

    x_data = pd.DataFrame(
        {"feature1": [1, 2, 3, 4, 5], "feature2": [0.5, 0.1, 0.9, 0.2, 0.4]}
    )
    y_data = pd.DataFrame({"income_encoded": [0, 0, 1, 0, 1]})

    return x_data, y_data


class TestDataSciencePipelineAutogluon:
    def test_evaluate_autogluon(self, sample_raw_data_auto_gluon):
        """
        Tests autogluon evaluation
        """

        x_data, y_data = sample_raw_data_auto_gluon

        mock_predictor = MagicMock()
        mock_predictor.label = "income_encoded"

        mock_metrics = {"accuracy": 0.85, "roc_auc": 0.92, "f1": 0.78}
        mock_predictor.evaluate.return_value = mock_metrics

        with patch(f"{NODES_PATH}.wandb") as mock_wandb:
            result = evaluate_autogluon(mock_predictor, x_data, y_data)

            assert isinstance(result, dict)

            assert "performance" in result

            metrics = result["performance"]

            assert metrics == mock_metrics

            for metric_name, value in metrics.items():
                assert 0.0 <= value <= 1.0, f"Metric {metric_name} outside of [0, 1]"

            mock_wandb.log.assert_called_once_with(mock_metrics)

    def test_model_directory_exists(self, sample_raw_data_auto_gluon):
        """
        Tests if directory exists.
        """

        model_path = Path("data/06_models")

        assert model_path.exists(), f"Directory {model_path} does not exist."
        assert model_path.is_dir(), f"Path {model_path} is not a directory."
