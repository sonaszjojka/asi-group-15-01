"""
This is a boilerplate pipeline 'data_science'
generated using Kedro 1.0.0
"""

from kedro.pipeline import Node, Pipeline

from .nodes import (
    start_wandb_run,
    load_raw,
    basic_clean,
    log_artifact,
    train_test_split,
    train_autogluon,
    evaluate_autogluon,
    save_best_model,
)


def create_pipeline(**kwargs) -> Pipeline:
    return Pipeline(
        [
            Node(
                func=start_wandb_run,
                inputs="params:wandb",
                outputs="wandb_started",
                name="start_wandb_run",
            ),
            Node(
                func=load_raw,
                inputs=["params:data.raw_csv", "wandb_started"],
                outputs="raw_data",
                name="load",
            ),
            Node(
                func=basic_clean,
                inputs="raw_data",
                outputs="cleaned_data",
                name="clean",
            ),
            Node(
                func=log_artifact,
                inputs=["cleaned_data", "params:artifacts.cleaned_data"],
                outputs="cleaned_data_logged",
                name="log_cleaned_data_artifact",
            ),
            Node(
                func=train_test_split,
                inputs=["cleaned_data_logged", "params:split"],
                outputs=[
                    "X_train",
                    "X_test",
                    "y_train",
                    "y_test",
                ],
                name="split",
            ),
            Node(
                func=train_autogluon,
                inputs=["X_train", "y_train", "params:autogluon", "params:seed"],
                outputs="ag_predictor",
                name="train_autogluon",
            ),
            Node(
                func=evaluate_autogluon,
                inputs=["ag_predictor", "X_test", "y_test"],
                outputs="ag_metrics",
                name="evaluate_autogluon",
            ),
            Node(
                func=save_best_model,
                inputs="ag_predictor",
                outputs="ag_model",
                name="save_best_model",
            ),
            Node(
                func=log_artifact,
                inputs=["ag_predictor", "params:artifacts.ag_model"],
                outputs=None,
                name="log_ag_model_artifact",
            ),
        ]
    )
