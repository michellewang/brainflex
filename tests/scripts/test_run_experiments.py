import importlib.util
import json
import sys
from enum import Enum
from pathlib import Path
from types import ModuleType
from unittest.mock import MagicMock

import numpy as np
import pandas as pd
import pytest
from fedbiomed.common.training_args import TrainingArgs
from fedbiomed.common.training_plans import BaseTrainingPlan
from pytest_mock import MockerFixture

FPATH_SCRIPT = Path(__file__).parents[2] / "scripts" / "run_experiments.py"
GLOBAL_STATS = {
    "mean": {
        "nb:Age": np.float64(30.0),
        "nb:Sex": np.float64(0.5),
        "lh_insula_thickness": np.float64(2.5),
        "Left-Hippocampus": np.float64(4000.0),
    },
    "std": {
        "nb:Age": np.float64(10.0),
        "nb:Sex": np.float64(0.5),
        "lh_insula_thickness": np.float64(0.5),
        "Left-Hippocampus": np.float64(100.0),
    },
}
NODE_METRICS = {
    "NODE_1": {
        "testing_global_updates": {
            "MEAN_SQUARE_ERROR": {1: {"iterations": [1], "values": [12.5]}}
        }
    }
}
TRAINING_REPLIES = {0: {"NODE_1": {"success": True, "sample_size": 80}}}


@pytest.fixture
def module(
    monkeypatch: pytest.MonkeyPatch, tmp_path_factory: pytest.TempPathFactory
) -> ModuleType:
    # importing fedbiomed.researcher creates a component directory
    monkeypatch.setenv(
        "FBM_RESEARCHER_COMPONENT_ROOT",
        str(tmp_path_factory.getbasetemp() / "fbm-researcher"),
    )
    # copy so that modules imported by the script are not kept
    monkeypatch.setattr(sys, "modules", dict(sys.modules))
    spec = importlib.util.spec_from_file_location("run_experiments", FPATH_SCRIPT)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def analytics() -> MagicMock:
    analytics = MagicMock()
    analytics.fetch_stats.return_value.global_stats.side_effect = GLOBAL_STATS.get
    return analytics


@pytest.fixture
def mock_experiment(
    module: ModuleType, mocker: MockerFixture, analytics: MagicMock
) -> MagicMock:
    mock_experiment = mocker.patch.object(module, "Experiment")
    exp = mock_experiment.return_value
    exp.analytics = analytics
    # mimic the getters of a real Experiment
    exp.model_args.side_effect = lambda: exp.set_model_args.call_args.args[0]
    exp.training_args.side_effect = lambda: TrainingArgs(
        mock_experiment.call_args.kwargs["training_args"]
    ).dict()
    exp.monitor.return_value._metric_store = NODE_METRICS
    exp.save_training_replies.return_value = TRAINING_REPLIES
    return mock_experiment


@pytest.fixture
def mock_run_task(module: ModuleType, mocker: MockerFixture) -> MagicMock:
    return mocker.patch.object(module, "_run_task")


@pytest.fixture
def fpath_csv(tmp_path: Path) -> Path:
    fpath = tmp_path / "dataset.csv"
    pd.DataFrame(
        {
            "nb:Age": [20.0, 30.0, 40.0],
            "nb:Sex": [0, 1, 1],
            "Left-Hippocampus": [4100.0, 4000.0, 3900.0],
            "lh_insula_thickness": [2.0, 2.5, 3.0],
        }
    ).to_csv(fpath, index=False)
    return fpath


class TestLoadTrainingPlanClass:
    @pytest.mark.parametrize("task", ["age", "sex"])
    def test_source(self, module: ModuleType, task: str):
        config = module.TASKS[task]
        plan_cls = module._load_training_plan_class(
            config.training_plan_module, config.training_plan_class
        )
        assert issubclass(plan_cls, BaseTrainingPlan)
        assert f"class {config.training_plan_class}" in plan_cls().source()


class TestGetFpathOut:
    def test_get_fpath_out(self, module: ModuleType, tmp_path: Path):
        fpath = module._get_fpath_out(
            tmp_path,
            "age",
            "results",
            n_rounds=10,
            batch_size=32,
            n_updates=50,
            learning_rate=0.001,
            test_ratio=0.2,
        )
        assert fpath == (
            tmp_path / "age" / "results-10rounds-50updates-32-0.001lr-0.2test.json"
        )

    def test_extension(self, module: ModuleType, tmp_path: Path):
        fpath = module._get_fpath_out(
            tmp_path, "sex", "model", 1, 1, 1, 0.1, 0.1, extension=".pkl"
        )
        assert fpath.parent == tmp_path / "sex"
        assert fpath.suffix == ".pkl"


class TestFilterFeatureCol:
    @pytest.mark.parametrize(
        "col",
        [
            "nb:Age",
            "nb:Sex",
            "lh_insula_thickness",
            "rh_precuneus_thickness",
            "EstimatedTotalIntraCranialVol",
            "Left-Hippocampus",
            "Right-Accumbens-area",
        ],
    )
    def test_included(self, module: ModuleType, col: str):
        assert module.filter_feature_col(col)

    @pytest.mark.parametrize(
        "col",
        [
            "lh_MeanThickness_thickness",
            "rh_MeanThickness_thickness",
            "rh_temporalpole_thickness",
            "Left-WM-hypointensities",
            "non-WM-hypointensities",
            "CSF",
            "5th-Ventricle",
            "lhSurfaceHoles",
            "lh_insula",
        ],
    )
    def test_excluded(self, module: ModuleType, col: str):
        assert not module.filter_feature_col(col)


class TestGetFeatureStats:
    def test_target_dropped(self, module: ModuleType, analytics: MagicMock):
        columns, mean, std, excluded_columns = module._get_feature_stats(
            analytics, "nb:Age"
        )
        assert columns == ["Left-Hippocampus", "lh_insula_thickness", "nb:Sex"]
        assert mean == [4000.0, 2.5, 0.5]
        assert std == [100.0, 0.5, 0.5]
        assert not excluded_columns
        analytics.fetch_stats.assert_called_once_with(["mean", "std"])

    def test_excluded_columns(self, module: ModuleType, analytics: MagicMock):
        stats = {
            "mean": GLOBAL_STATS["mean"]
            | {
                "Left-Thalamus": 0.0,
                "lh_cuneus_thickness": 1.0,
                "lhSurfaceHoles": 20.0,
            },
            "std": GLOBAL_STATS["std"]
            | {
                "Left-Thalamus": 0.0,
                "lh_cuneus_thickness": np.nan,
                "lhSurfaceHoles": 5.0,
            },
        }
        analytics.fetch_stats.return_value.global_stats.side_effect = stats.get
        columns, mean, std, excluded_columns = module._get_feature_stats(
            analytics, "nb:Age"
        )
        assert columns == ["Left-Hippocampus", "lh_insula_thickness", "nb:Sex"]
        assert len(mean) == len(std) == 3
        assert excluded_columns == {
            "not in feature column list": ["lhSurfaceHoles"],
            "zero or infinite standard deviation": [
                "Left-Thalamus",
                "lh_cuneus_thickness",
            ],
        }

    def test_json_serializable(self, module: ModuleType, analytics: MagicMock):
        json.dumps(module._get_feature_stats(analytics, "nb:Sex"))


class TestBuildModelArgs:
    @pytest.mark.parametrize(
        "task,target_column,expected_x",
        [
            ("age", "nb:Age", [[1.0, -1.0, -1.0], [0.0, 0.0, 1.0], [-1.0, 1.0, 1.0]]),
            ("sex", "nb:Sex", [[1.0, -1.0, -1.0], [0.0, 0.0, 0.0], [-1.0, 1.0, 1.0]]),
        ],
    )
    def test_training_data(
        self,
        module: ModuleType,
        analytics: MagicMock,
        fpath_csv: Path,
        task: str,
        target_column: str,
        expected_x: list[list[float]],
    ):
        config = module.TASKS[task]
        columns, mean, std, _ = module._get_feature_stats(analytics, target_column)
        model_args = module._build_model_args(task, columns, mean, std, 0.1)

        plan = module._load_training_plan_class(
            config.training_plan_module, config.training_plan_class
        )()
        plan.post_init(model_args, TrainingArgs({}, only_required=False))
        data_manager = plan.training_data()
        data_manager.load(tp_type=plan.type())
        data_manager.load_dataset(root=str(fpath_csv))
        dataset = data_manager._dataset
        np.testing.assert_allclose(
            [dataset[i][0] for i in range(len(dataset))], expected_x
        )

    def test_extra_model_args(self, module: ModuleType):
        model_args = module._build_model_args("sex", ["a"], [0.0], [1.0], 0.1)
        assert model_args["n_classes"] == 2
        assert model_args["eta0"] == 0.1


class TestSerialize:
    @pytest.mark.parametrize(
        "obj,expected",
        [
            (Enum("Color", "RED").RED, "RED"),
            (Path("a/b"), "a/b"),
            (np.array([1.0, 2.0]), [1.0, 2.0]),
            (np.float64(1.5), 1.5),
            ({"b", "a"}, ["a", "b"]),
        ],
    )
    def test_serialize(self, module: ModuleType, obj: object, expected: object):
        assert module._serialize(obj) == expected

    def test_not_serializable(self, module: ModuleType):
        with pytest.raises(TypeError, match="not serializable"):
            module._serialize(object())


class TestRunTask:
    def test_run_task(
        self, module: ModuleType, mock_experiment: MagicMock, tmp_path: Path
    ):
        settings = {"n_rounds": 3}
        fpath_results = tmp_path / "age" / "results.json"
        fpath_model = tmp_path / "age" / "model.pkl"
        module._run_task(
            "age",
            n_rounds=3,
            batch_size=8,
            n_updates=10,
            learning_rate=0.1,
            test_ratio=0.2,
            settings=settings,
            fpath_results=fpath_results,
            fpath_model=fpath_model,
        )

        assert mock_experiment.call_args.kwargs["tags"] == ["brainflex", "age"]
        assert mock_experiment.call_args.kwargs["round_limit"] == 3
        training_args = TrainingArgs(mock_experiment.call_args.kwargs["training_args"])
        assert training_args["loader_args"] == {"batch_size": 8}
        assert training_args["num_updates"] == 10
        exp = mock_experiment.return_value
        assert exp.run_once.call_count == 3
        assert exp.save_training_replies.call_count == 3
        exp.training_plan.return_value.export_model.assert_called_once_with(
            str(fpath_model)
        )
        assert (
            exp.set_training_plan_class.call_args.args[0].__name__
            == "TabularRegressorTrainingPlan"
        )

        json_data = json.loads(fpath_results.read_text())
        assert json_data["settings"]["runtime_args"] == settings
        assert json_data["settings"]["excluded_columns"] == {}
        model_args = json_data["settings"]["model_args"]
        assert model_args["target_column"] == "nb:Age"
        assert model_args["eta0"] == 0.1
        assert model_args["feature_columns"] == [
            "Left-Hippocampus",
            "lh_insula_thickness",
            "nb:Sex",
        ]
        exp.set_model_args.assert_called_once_with(model_args)
        training_args = json_data["settings"]["training_args"]
        assert training_args["test_metric"] == "MEAN_SQUARE_ERROR"
        # JSON object keys are strings
        assert json_data["results"] == json.loads(
            json.dumps(
                {"node_metrics": NODE_METRICS, "training_replies": TRAINING_REPLIES}
            )
        )

    def test_settings_saved_if_training_fails(
        self, module: ModuleType, mock_experiment: MagicMock, tmp_path: Path
    ):
        mock_experiment.return_value.run_once.side_effect = RuntimeError
        fpath_results = tmp_path / "sex" / "results.json"
        with pytest.raises(RuntimeError):
            module._run_task(
                "sex",
                n_rounds=3,
                batch_size=8,
                n_updates=10,
                learning_rate=0.1,
                test_ratio=0.2,
                settings={},
                fpath_results=fpath_results,
                fpath_model=tmp_path / "sex" / "model.pkl",
            )
        json_data = json.loads(fpath_results.read_text())
        assert json_data["settings"]["model_args"]["n_classes"] == 2
        assert "results" not in json_data


def _get_fpath_model(module: ModuleType, dpath: Path, task: str) -> Path:
    """Model path for the default CLI arguments."""
    return module._get_fpath_out(
        dpath,
        task,
        module.PREFIX_MODEL,
        n_rounds=module.DEFAULT_N_ROUNDS,
        batch_size=module.DEFAULT_BATCH_SIZE,
        n_updates=module.DEFAULT_N_UPDATES,
        learning_rate=module.DEFAULT_LEARNING_RATE,
        test_ratio=module.DEFAULT_TEST_RATIO,
        extension=".pkl",
    )


class TestMain:
    def test_tasks_in_order(
        self, module: ModuleType, mock_run_task: MagicMock, tmp_path: Path
    ):
        module.main(
            [str(tmp_path), "--task", "sex", "--task", "age", "--n-rounds", "2"]
        )
        assert [call.kwargs["task"] for call in mock_run_task.call_args_list] == [
            "sex",
            "age",
        ]
        kwargs = mock_run_task.call_args.kwargs
        assert kwargs["n_rounds"] == 2
        assert kwargs["fpath_results"].parent == tmp_path.resolve() / "age"
        assert kwargs["fpath_results"].name.startswith("results-2rounds-")
        assert kwargs["fpath_model"].name.startswith("model-2rounds-")
        assert kwargs["fpath_model"].suffix == ".pkl"
        settings = kwargs["settings"]
        assert settings["output_dir"] == tmp_path.resolve()
        assert settings["tasks"] == ["sex", "age"]
        assert settings["n_rounds"] == 2
        assert settings["learning_rate"] == module.DEFAULT_LEARNING_RATE
        assert settings["overwrite"] is False

    def test_existing_model(
        self, module: ModuleType, mock_run_task: MagicMock, tmp_path: Path
    ):
        fpath_model = _get_fpath_model(module, tmp_path.resolve(), "age")
        fpath_model.parent.mkdir()
        fpath_model.touch()
        with pytest.raises(SystemExit, match="--overwrite"):
            module.main([str(tmp_path), "--task", "sex", "--task", "age"])
        mock_run_task.assert_not_called()

    def test_overwrite(
        self, module: ModuleType, mock_run_task: MagicMock, tmp_path: Path
    ):
        fpath_model = _get_fpath_model(module, tmp_path.resolve(), "age")
        fpath_model.parent.mkdir()
        fpath_model.touch()
        module.main([str(tmp_path), "--task", "age", "--overwrite"])
        mock_run_task.assert_called_once()

    @pytest.mark.parametrize("args", [[], ["--task", "height"]])
    def test_invalid_task(
        self,
        module: ModuleType,
        mock_run_task: MagicMock,
        tmp_path: Path,
        args: list[str],
    ):
        with pytest.raises(SystemExit):
            module.main([str(tmp_path), *args])
        mock_run_task.assert_not_called()
