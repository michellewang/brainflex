#!/usr/bin/env python3

"""Run the federated learning experiments."""

import argparse
import importlib.util
import json
import math
import sys
from collections import defaultdict
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path

import numpy as np
from fedbiomed.common.metrics import MetricTypes
from fedbiomed.common.training_plans import BaseTrainingPlan
from fedbiomed.researcher.federated_workflows import Experiment, FederatedAnalytics

DPATH_TRAINING_PLANS = Path(__file__).resolve().parents[1] / "training_plans"
PREFIX_MODEL = "model"
PREFIX_RESULTS = "results"
COMMON_TAG = "brainflex"

DEFAULT_N_ROUNDS = 10
DEFAULT_BATCH_SIZE = 32
DEFAULT_N_UPDATES = 50
DEFAULT_LEARNING_RATE = 0.001
DEFAULT_TEST_RATIO = 0.1


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
        "--n-rounds",
        type=int,
        default=DEFAULT_N_ROUNDS,
        help=f"Number of FL rounds. Default: {DEFAULT_N_ROUNDS}.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=DEFAULT_BATCH_SIZE,
        help=f"Local batch size. Default: {DEFAULT_BATCH_SIZE}.",
    )
    parser.add_argument(
        "--n-updates",
        type=int,
        default=DEFAULT_N_UPDATES,
        help=f"Local updates per round. Default: {DEFAULT_N_UPDATES}.",
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=DEFAULT_LEARNING_RATE,
        help=f"SGD learning rate. Default: {DEFAULT_LEARNING_RATE}.",
    )
    parser.add_argument(
        "--test-ratio",
        type=float,
        default=DEFAULT_TEST_RATIO,
        help=f"Fraction of node data for validation. Default: {DEFAULT_TEST_RATIO}.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing results.",
    )
    return parser


def _get_fpath_out(
    output_dir: Path,
    task: str,
    prefix: str,
    n_rounds: int,
    batch_size: int,
    n_updates: int,
    learning_rate: float,
    test_ratio: float,
    extension: str = ".json",
) -> Path:
    stem = "-".join(
        [
            prefix,
            f"{n_rounds}rounds",
            f"{n_updates}updates",
            f"{batch_size}",
            f"{learning_rate}lr",
            f"{test_ratio}test",
        ]
    )
    return output_dir / task / f"{stem}{extension}"


def _filter_feature_col(col: str) -> bool:
    if col.startswith("nb:"):
        return True
    elif col.endswith("_thickness"):
        return not (
            col
            in [
                # exclude these
                "lh_MeanThickness_thickness",
                "rh_MeanThickness_thickness",
                "rh_temporalpole_thickness",
            ]
        )
    else:
        return col in [
            # include these
            "EstimatedTotalIntraCranialVol",
            "Left-Lateral-Ventricle",
            "Left-Thalamus",
            "Left-Caudate",
            "Left-Putamen",
            "Left-Pallidum",
            "Left-Hippocampus",
            "Left-Amygdala",
            "Left-Accumbens-area",
            "Right-Lateral-Ventricle",
            "Right-Thalamus",
            "Right-Caudate",
            "Right-Putamen",
            "Right-Pallidum",
            "Right-Hippocampus",
            "Right-Amygdala",
            "Right-Accumbens-area",
        ]


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
) -> tuple[list[str], list[float], list[float], dict[str, list[str]]]:
    """Get the feature columns and their global mean and std."""
    result = analytics.fetch_stats(["mean", "std"])
    mean = result.global_stats("mean")
    std = result.global_stats("std")
    excluded_columns: dict[str, list[str]] = defaultdict(list)
    columns: list[str] = []
    for column in mean:
        if not _filter_feature_col(column):
            excluded_columns["not in feature column list"].append(column)
            continue
        if not math.isfinite(std[column]) or std[column] == 0:
            excluded_columns["zero or infinite standard deviation"].append(column)
            continue
        if column != target_column:
            columns.append(column)

    columns = sorted(columns)

    if excluded_columns:
        for reason, cols in excluded_columns.items():
            print(f"Excluding {len(cols)} features ({reason}): {cols}")

    # cast to float so that the model args are JSON-serializable
    return (
        columns,
        [float(mean[column]) for column in columns],
        [float(std[column]) for column in columns],
        excluded_columns,
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
    n_rounds: int,
    batch_size: int,
    n_updates: int,
    learning_rate: float,
    test_ratio: float,
    settings: dict,
    fpath_results: Path,
    fpath_model: Path,
) -> BaseTrainingPlan:
    """Train a model for a task and save it in a task-specific output directory."""
    config = TASKS[task]
    exp = Experiment(
        tags=[COMMON_TAG, task],
        training_args={
            "loader_args": {"batch_size": batch_size},
            "num_updates": n_updates,
            "test_ratio": test_ratio,
            "test_metric": config.test_metric,
            "test_on_global_updates": True,
            "test_on_local_updates": True,
            # NOTE: if Leave-one-node-out CV, set shuffle_testing_dataset=True
        },
        round_limit=n_rounds,
    )

    # the training plan needs the feature stats to be instantiated
    columns, mean, std, excluded_columns = _get_feature_stats(
        exp.analytics, config.target_column
    )
    print(f"[{task}] Using {len(columns)} features")
    model_args = _build_model_args(task, columns, mean, std, learning_rate)
    exp.set_model_args(model_args)
    exp.set_training_plan_class(
        _load_training_plan_class(
            config.training_plan_module, config.training_plan_class
        )
    )

    # saved before training so that they are available even if training fails
    json_data = {
        "settings": {
            "runtime_args": settings,
            "excluded_columns": excluded_columns,
            "model_args": exp.model_args(),
            "training_args": exp.training_args(),
        }
    }
    fpath_results.parent.mkdir(parents=True, exist_ok=True)
    _save_json(fpath_results, json_data)

    for round_ in range(1, n_rounds + 1):
        print(f"[{task}] Round {round_}/{n_rounds}")
        exp.run_once()
        json_data["results"] = {
            # node training losses and test metrics (no public getter)
            "node_metrics": exp.monitor()._metric_store,
            "training_replies": exp.save_training_replies(),
        }
        _save_json(fpath_results, json_data)
    print(f"[{task}] Saved settings and results to {fpath_results}")

    fpath_model.parent.mkdir(parents=True, exist_ok=True)
    exp.training_plan().export_model(str(fpath_model))
    print(f"[{task}] Saved global model to {fpath_model}")
    return exp.training_plan()


def run_experiments(
    output_dir: Path,
    tasks: list[str],
    n_rounds: int = DEFAULT_N_ROUNDS,
    batch_size: int = DEFAULT_BATCH_SIZE,
    n_updates: int = DEFAULT_N_UPDATES,
    learning_rate: float = DEFAULT_LEARNING_RATE,
    test_ratio: float = DEFAULT_TEST_RATIO,
    overwrite: bool = False,
) -> None:
    """Train and save a global model for each task."""
    output_dir = output_dir.resolve()
    settings = locals().copy()

    fpaths_results = []
    fpaths_model = []
    for task in tasks:
        fpaths_results.append(
            _get_fpath_out(
                output_dir=output_dir,
                task=task,
                prefix=PREFIX_RESULTS,
                n_rounds=n_rounds,
                batch_size=batch_size,
                n_updates=n_updates,
                learning_rate=learning_rate,
                test_ratio=test_ratio,
                extension=".json",
            )
        )
        fpaths_model.append(
            _get_fpath_out(
                output_dir=output_dir,
                task=task,
                prefix=PREFIX_MODEL,
                n_rounds=n_rounds,
                batch_size=batch_size,
                n_updates=n_updates,
                learning_rate=learning_rate,
                test_ratio=test_ratio,
                extension=".pkl",
            )
        )

    # error out if one of the result files already exists
    fpaths_results_existing = [
        fpath
        for fpath in fpaths_results
        if fpath.exists() and "results" in json.loads(fpath.read_text())
    ]
    if fpaths_results_existing and not overwrite:
        raise FileExistsError(
            f"Results already exist: {[str(fpath) for fpath in fpaths_results_existing]}."
            " Use --overwrite to overwrite them"
        )

    for task, fpath_results, fpath_model in zip(tasks, fpaths_results, fpaths_model):
        _run_task(
            task=task,
            n_rounds=n_rounds,
            batch_size=batch_size,
            n_updates=n_updates,
            learning_rate=learning_rate,
            test_ratio=test_ratio,
            settings=settings,
            fpath_results=fpath_results,
            fpath_model=fpath_model,
        )


def main(argv: list[str] | None = None) -> None:
    args = _get_parser().parse_args(argv)
    try:
        run_experiments(
            args.output_dir,
            args.tasks,
            n_rounds=args.n_rounds,
            batch_size=args.batch_size,
            n_updates=args.n_updates,
            learning_rate=args.learning_rate,
            test_ratio=args.test_ratio,
            overwrite=args.overwrite,
        )
    except FileExistsError as e:
        sys.exit(f"Error: {e}")


if __name__ == "__main__":
    main()
