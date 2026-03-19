import io
from pathlib import Path
from types import SimpleNamespace

import h5py
import numpy as np

from brainsets_pipelines.neuroprobe_2025 import pipeline as neuroprobe_pipeline


def test_extract_neural_data_uses_sample_index_timestamps(tmp_path):
    input_file = tmp_path / "sub_1_trial001.h5"
    with h5py.File(input_file, "w") as handle:
        data_group = handle.create_group("data")
        data_group.create_dataset("electrode_1", data=np.array([1.0, 2.0, 3.0]))
        data_group.create_dataset("electrode_2", data=np.array([4.0, 5.0, 6.0]))

    class _Channels:
        h5_label = np.array(["electrode_1", "electrode_2"], dtype=np.str_)

        def __len__(self):
            return len(self.h5_label)

    neuroprobe_pipeline.neuroprobe_config = SimpleNamespace(SAMPLING_RATE=2.0)
    seeg_data = neuroprobe_pipeline._extract_neural_data(input_file, _Channels())

    np.testing.assert_allclose(
        np.asarray(seeg_data.timestamps), np.array([0.0, 0.5, 1.0], dtype=np.float64)
    )


def test_download_file_applies_timeout_and_retries(tmp_path, monkeypatch):
    destination = tmp_path / "artifact.bin"
    payload = b"ok"
    seen_timeouts = []
    calls = {"count": 0}

    class _FakeResponse(io.BytesIO):
        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            self.close()
            return False

    def _fake_urlopen(url, timeout=None):
        seen_timeouts.append(timeout)
        calls["count"] += 1
        if calls["count"] < 3:
            raise TimeoutError("stalled")
        return _FakeResponse(payload)

    monkeypatch.setattr(neuroprobe_pipeline.urllib.request, "urlopen", _fake_urlopen)
    monkeypatch.setattr(neuroprobe_pipeline, "DOWNLOAD_TIMEOUT_SECONDS", 7)
    monkeypatch.setattr(neuroprobe_pipeline, "DOWNLOAD_MAX_RETRIES", 2)
    monkeypatch.setattr(neuroprobe_pipeline, "DOWNLOAD_RETRY_BACKOFF_SECONDS", 0.0)
    monkeypatch.setattr(neuroprobe_pipeline.time, "sleep", lambda _seconds: None)

    neuroprobe_pipeline._download_file(
        "https://example.com/artifact.bin",
        destination,
        overwrite=True,
    )

    assert destination.read_bytes() == payload
    assert calls["count"] == 3
    assert seen_timeouts == [7, 7, 7]
    assert not destination.with_suffix(".bin.tmp").exists()


def test_process_prepares_worker_runtime_even_when_processing_is_skipped(tmp_path):
    pipeline_instance = neuroprobe_pipeline.Pipeline.__new__(
        neuroprobe_pipeline.Pipeline
    )
    pipeline_instance.raw_dir = tmp_path / "raw"
    pipeline_instance.processed_dir = tmp_path / "processed"
    pipeline_instance.processed_dir.mkdir(parents=True, exist_ok=True)
    pipeline_instance.args = SimpleNamespace(reprocess=False, no_splits=False)
    pipeline_instance.update_status = lambda _status: None

    input_file = tmp_path / "sub_1_trial001.h5"
    input_file.touch()
    (pipeline_instance.processed_dir / input_file.name).touch()

    prepared = []
    pipeline_instance._prepare_worker_runtime = lambda: prepared.append(True)

    neuroprobe_pipeline.Pipeline.process(pipeline_instance, input_file)
    assert prepared == [True]


def test_iterate_extract_splits_prepares_and_deduplicates_subject_initialization(
    tmp_path, monkeypatch
):
    pipeline_instance = neuroprobe_pipeline.Pipeline.__new__(
        neuroprobe_pipeline.Pipeline
    )
    pipeline_instance.raw_dir = tmp_path / "raw"
    pipeline_instance.update_status = lambda _status: None

    prepared_raw_dirs = []
    created_subject_ids = []
    seen_subject_keys = []

    class _FakeNeuroprobe:
        class BrainTreebankSubject:
            def __init__(self, *, subject_id, **_kwargs):
                created_subject_ids.append(subject_id)
                self.subject_id = subject_id

    def _fake_prepare(raw_dir: Path) -> None:
        prepared_raw_dirs.append(raw_dir)
        neuroprobe_pipeline.neuroprobe = _FakeNeuroprobe
        neuroprobe_pipeline.neuroprobe_config = SimpleNamespace(
            NEUROPROBE_FULL_SUBJECT_TRIALS=[(1, 0), (1, 1), (2, 0)],
            NEUROPROBE_LITE_SUBJECT_TRIALS=set(),
            NEUROPROBE_NANO_SUBJECT_TRIALS=set(),
        )

    def _fake_extract_and_structure_splits(**kwargs):
        seen_subject_keys.append(sorted(kwargs["all_subjects"].keys()))
        return {"split_key": object()}

    monkeypatch.setattr(neuroprobe_pipeline, "_prepare_neuroprobe_lib", _fake_prepare)
    monkeypatch.setattr(
        neuroprobe_pipeline,
        "ALL_EVAL_SETTINGS",
        {
            "lite": [False],
            "nano": [False],
            "binary_tasks": [True],
            "eval_setting": ["within_session"],
        },
    )
    monkeypatch.setattr(
        neuroprobe_pipeline,
        "_extract_channel_data",
        lambda subject: f"channels-{subject.subject_id}",
    )
    monkeypatch.setattr(
        neuroprobe_pipeline,
        "_extract_and_structure_splits",
        _fake_extract_and_structure_splits,
    )

    split_indices = neuroprobe_pipeline.Pipeline.iterate_extract_splits(
        pipeline_instance,
        subject_id=1,
        trial_id=0,
    )

    assert prepared_raw_dirs == [pipeline_instance.raw_dir]
    assert created_subject_ids == [1, 2]
    assert seen_subject_keys == [[1, 2]]
    assert "split_key" in split_indices
