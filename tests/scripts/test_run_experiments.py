import importlib.util
import json
import sys
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
        "lh_insula": np.float64(2.5),
        "Left-Hippocampus": np.float64(4000.0),
    },
    "std": {
        "nb:Age": np.float64(10.0),
        "nb:Sex": np.float64(0.5),
        "lh_insula": np.float64(0.5),
        "Left-Hippocampus": np.float64(100.0),
    },
}


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
    mock_experiment.return_value.analytics = analytics
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
            "lh_insula": [2.0, 2.5, 3.0],
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


class TestGetFeatureStats:
    def test_target_dropped(self, module: ModuleType, analytics: MagicMock):
        columns, mean, std = module._get_feature_stats(analytics, "nb:Age")
        assert columns == ["Left-Hippocampus", "lh_insula", "nb:Sex"]
        assert mean == [4000.0, 2.5, 0.5]
        assert std == [100.0, 0.5, 0.5]
        analytics.fetch_stats.assert_called_once_with(["mean", "std"])

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
        columns, mean, std = module._get_feature_stats(analytics, target_column)
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


class TestRunTask:
    def test_run_task(
        self, module: ModuleType, mock_experiment: MagicMock, tmp_path: Path
    ):
        module._run_task(
            "age",
            tmp_path,
            rounds=3,
            batch_size=8,
            num_updates=10,
            learning_rate=0.1,
            test_ratio=0.2,
        )

        assert mock_experiment.call_args.kwargs["tags"] == ["brainflex", "age"]
        assert mock_experiment.call_args.kwargs["round_limit"] == 3
        exp = mock_experiment.return_value
        assert exp.run_once.call_count == 3
        exp.training_plan.return_value.export_model.assert_called_once_with(
            str(tmp_path / "age" / "model.pkl")
        )
        model_args = json.loads((tmp_path / "age" / "model_args.json").read_text())
        assert model_args["target_column"] == "nb:Age"
        assert model_args["feature_columns"] == [
            "Left-Hippocampus",
            "lh_insula",
            "nb:Sex",
        ]
        exp.set_model_args.assert_called_once_with(model_args)
        assert (
            exp.set_training_plan_class.call_args.args[0].__name__
            == "TabularRegressorTrainingPlan"
        )


class TestMain:
    def test_tasks_in_order(
        self, module: ModuleType, mock_run_task: MagicMock, tmp_path: Path
    ):
        module.main([str(tmp_path), "--task", "sex", "--task", "age", "--rounds", "2"])
        assert [call.args[0] for call in mock_run_task.call_args_list] == [
            "sex",
            "age",
        ]
        assert mock_run_task.call_args.kwargs["rounds"] == 2

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
