#!/usr/bin/env python3

"""Run the federated learning experiments."""

import argparse
import importlib.util
import json
import math
import sys
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path

import numpy as np
from fedbiomed.common.metrics import MetricTypes
from fedbiomed.common.training_plans import BaseTrainingPlan
from fedbiomed.researcher.federated_workflows import Experiment, FederatedAnalytics

DPATH_TRAINING_PLANS = Path(__file__).resolve().parents[1] / "training_plans"
FNAME_MODEL = "model.pkl"
FNAME_RESULTS = "results.json"
COMMON_TAG = "brainflex"


@dataclass(frozen=True)
class TaskConfig:
    training_plan_module: str
    training_plan_class: str
    target_column: str
    test_metric: MetricTypes
    extra_model_args: dict = field(default_factory=dict)


TASKS = {
    "age": TaskConfig(
        training_plan_module="tabular_regressor",
        training_plan_class="TabularRegressorTrainingPlan",
        target_column="nb:Age",
        test_metric=MetricTypes.MEAN_SQUARE_ERROR,
    ),
    "sex": TaskConfig(
        training_plan_module="tabular_classifier",
        training_plan_class="TabularClassifierTrainingPlan",
        target_column="nb:Sex",
        test_metric=MetricTypes.ACCURACY,
        extra_model_args={"n_classes": 2},
    ),
}


def _get_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "output_dir", type=Path, help="Directory to save the global models to."
    )
    parser.add_argument(
        "--task",
        dest="tasks",
        action="append",
        required=True,
        choices=list(TASKS),
        help="Task to run. Can be repeated.",
    )
    parser.add_argument(
        "--rounds", type=int, default=10, help="Number of FL rounds. Default: 10."
    )
    parser.add_argument(
        "--batch-size", type=int, default=32, help="Local batch size. Default: 32."
    )
    parser.add_argument(
        "--num-updates",
        type=int,
        default=50,
        help="Local updates per round. Default: 50.",
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=0.01,
        help="SGD learning rate. Default: 0.01.",
    )
    parser.add_argument(
        "--test-ratio",
        type=float,
        default=0.2,
        help="Fraction of node data for validation. Default: 0.2.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing results.",
    )
    return parser


def _serialize(obj: object) -> object:
    if isinstance(obj, Enum):
        return obj.name
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, (np.ndarray, np.generic)):
        return obj.tolist()
    if isinstance(obj, set):
        return sorted(obj)
    raise TypeError(f"Type {type(obj)} not serializable")


def _save_json(fpath: Path, data: dict) -> None:
    fpath.write_text(
        json.dumps(data, indent=4, allow_nan=False, default=_serialize) + "\n"
    )


def _load_training_plan_class(module_name: str, class_name: str) -> type:
    spec = importlib.util.spec_from_file_location(
        f"training_plans.{module_name}",
        DPATH_TRAINING_PLANS / f"{module_name}.py",
    )
    module = importlib.util.module_from_spec(spec)
    # registered so that Fed-BioMed can get the source code to send to nodes
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return getattr(module, class_name)


def _get_feature_stats(
    analytics: FederatedAnalytics, target_column: str
) -> tuple[list[str], list[float], list[float]]:
    """Get the feature columns and their global mean and std."""
    result = analytics.fetch_stats(["mean", "std"])
    mean = result.global_stats("mean")
    std = result.global_stats("std")
    excluded_columns: list[str] = []
    columns: list[str] = []
    for column in mean:
        if not math.isfinite(std[column]) or std[column] == 0:
            excluded_columns.append(column)
            continue
        if column != target_column:
            columns.append(column)

    columns = sorted(columns)

    if excluded_columns:
        print(f"Excluding {len(excluded_columns)} features: {excluded_columns}")

    # cast to float so that the model args are JSON-serializable
    return (
        columns,
        [float(mean[column]) for column in columns],
        [float(std[column]) for column in columns],
    )


def _build_model_args(
    task: str,
    columns: list[str],
    mean: list[float],
    std: list[float],
    learning_rate: float,
) -> dict:
    config = TASKS[task]
    return {
        "n_features": len(columns),
        "feature_columns": columns,
        "feature_mean": mean,
        "feature_std": std,
        "target_column": config.target_column,
        "eta0": learning_rate,
        **config.extra_model_args,
    }


def _run_task(
    task: str,
    output_dir: Path,
    rounds: int,
    batch_size: int,
    num_updates: int,
    learning_rate: float,
    test_ratio: float,
    settings: dict,
) -> BaseTrainingPlan:
    """Train a model for a task and save it in a task-specific output directory."""
    config = TASKS[task]
    exp = Experiment(
        tags=[COMMON_TAG, task],
        training_args={
            "loader_args": {"batch_size": batch_size},
            "num_updates": num_updates,
            "test_ratio": test_ratio,
            "test_metric": config.test_metric,
            "test_on_global_updates": True,
            "test_on_local_updates": True,
            # NOTE: if Leave-one-node-out CV, set shuffle_testing_dataset=True
        },
        round_limit=rounds,
    )

    # the training plan needs the feature stats to be instantiated
    columns, mean, std = _get_feature_stats(exp.analytics, config.target_column)
    print(f"[{task}] Using {len(columns)} features")
    model_args = _build_model_args(task, columns, mean, std, learning_rate)
    exp.set_model_args(model_args)
    exp.set_training_plan_class(
        _load_training_plan_class(
            config.training_plan_module, config.training_plan_class
        )
    )

    dpath_task = output_dir / task
    dpath_task.mkdir(parents=True, exist_ok=True)
    fpath_results = dpath_task / FNAME_RESULTS
    # saved before training so that they are available even if training fails
    json_data = {
        "settings": {
            "runtime_args": settings,
            "model_args": exp.model_args(),
            "training_args": exp.training_args(),
        }
    }
    _save_json(fpath_results, json_data)

    for round_ in range(1, rounds + 1):
        print(f"[{task}] Round {round_}/{rounds}")
        exp.run_once()
        json_data["results"] = {
            # node training losses and test metrics (no public getter)
            "node_metrics": exp.monitor()._metric_store,
            "training_replies": exp.save_training_replies(),
        }
        _save_json(fpath_results, json_data)
    print(f"[{task}] Saved settings and results to {fpath_results}")

    fpath_model = dpath_task / FNAME_MODEL
    exp.training_plan().export_model(str(fpath_model))
    print(f"[{task}] Saved global model to {fpath_model}")
    return exp.training_plan()


def run_experiments(
    output_dir: Path,
    tasks: list[str],
    rounds: int = 10,
    batch_size: int = 32,
    num_updates: int = 50,
    learning_rate: float = 0.01,
    test_ratio: float = 0.2,
    overwrite: bool = False,
) -> None:
    """Train and save a global model for each task."""
    settings = {
        "output_dir": output_dir,
        "tasks": tasks,
        "rounds": rounds,
        "batch_size": batch_size,
        "num_updates": num_updates,
        "learning_rate": learning_rate,
        "test_ratio": test_ratio,
        "overwrite": overwrite,
    }
    
    # error out if one of the result files already exists
    fpaths_existing = [
        fpath for task in tasks if (fpath := output_dir / task / FNAME_RESULTS).exists()
    ]
    if fpaths_existing and not overwrite:
        raise FileExistsError(
            f"Results already exist: {[str(fpath) for fpath in fpaths_existing]}."
            " Use --overwrite to overwrite them"
        )

    for task in tasks:
        _run_task(
            task,
            output_dir,
            rounds=rounds,
            batch_size=batch_size,
            num_updates=num_updates,
            learning_rate=learning_rate,
            test_ratio=test_ratio,
            settings=settings,
        )


def main(argv: list[str] | None = None) -> None:
    args = _get_parser().parse_args(argv)
    try:
        run_experiments(
            args.output_dir,
            args.tasks,
            rounds=args.rounds,
            batch_size=args.batch_size,
            num_updates=args.num_updates,
            learning_rate=args.learning_rate,
            test_ratio=args.test_ratio,
            overwrite=args.overwrite,
        )
    except FileExistsError as e:
        sys.exit(f"Error: {e}")


if __name__ == "__main__":
    main()
