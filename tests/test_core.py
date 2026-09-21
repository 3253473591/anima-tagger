from __future__ import annotations

import json
import sys
import types
from datetime import datetime, timedelta
from pathlib import Path

import pytest
from PIL import Image

from anima_tagger.application.review import ReviewService
from anima_tagger.application import controller as controller_module
from anima_tagger.domain.errors import CaptionValidationError
from anima_tagger.domain.models import LoraMode
from anima_tagger.domain.models import ImageAsset, JobStatus
from anima_tagger.domain.validation import normalize_first_caption, validate_caption, validate_first_caption, validate_second_review
from anima_tagger.infrastructure.files import atomic_write, is_nonempty, scan_images, stage_path
from anima_tagger.infrastructure.local_inference import LocalInferenceAdapter
from anima_tagger.infrastructure.repository import Repository, now
from anima_tagger.prompts import load_prompt
from anima_tagger.domain.models import Stage


def make_asset(tmp_path: Path):
    image = tmp_path / "sample.png"
    Image.new("RGB", (16, 16), "red").save(image)
    repo = Repository(tmp_path / "state.sqlite3")
    folder_id = repo.register_folder(tmp_path)
    assets = scan_images(tmp_path, folder_id)
    repo.upsert_images(assets)
    return repo, assets[0]


def test_prompt_assets_load_and_substitute_placeholders():
    """The published repository ships prompt skeletons, not production prompt assets.

    The length of the asset is therefore not an invariant here. What must hold is that
    every mode/stage resolves to an existing asset, that the skeleton is non-trivial, and
    that neither placeholder leaks into the request sent to the model.
    """
    for mode in LoraMode:
        first = load_prompt(mode, Stage.FIRST_REVIEW, trigger_word="TriggerX")
        second = load_prompt(mode, Stage.SECOND_REVIEW, trigger_word="TriggerX", first_caption="TriggerX, 1girl")
        assert len(first) > 400 and len(second) > 400
        assert "{trigger_word}" not in first + second
        assert "{first_caption}" not in first + second
        assert "TriggerX" in first and "TriggerX" in second
        assert "TriggerX, 1girl" in second


def test_caption_validation():
    parsed = validate_caption("TriggerX, 1girl, blue hair, looking at viewer", "TriggerX", LoraMode.CHARACTER)
    assert parsed.tags[0] == "TriggerX"
    with pytest.raises(CaptionValidationError):
        validate_caption("TriggerX, 1girl, masterpiece", "TriggerX", LoraMode.CHARACTER)
    with pytest.raises(CaptionValidationError):
        validate_caption("wrong, 1girl", "TriggerX", LoraMode.CHARACTER)


def test_first_caption_accepts_format_issues_and_can_filter_them():
    caption = "TriggerX, 1Girl, 1girl, masterpiece, blue hair, (smile:1.2)"
    assert validate_first_caption(caption, "TriggerX", LoraMode.CHARACTER).tags[1] == "1Girl"
    assert normalize_first_caption(caption, "TriggerX", LoraMode.CHARACTER) == "TriggerX, 1girl, blue hair"


def test_second_review_contract():
    result = {
        "schema_version": 1, "score": 9, "comment": "ok",
        "final_caption": "TriggerX, 1girl, blue hair",
        "first_caption_validity": [{"tag": "1girl", "valid": True, "suggestion": None}],
        "tag_translations": [
            {"tag": "TriggerX", "cn": "TriggerX"},
            {"tag": "1girl", "cn": "1名女孩"},
            {"tag": "blue hair", "cn": "蓝色头发"},
        ],
        "first_review_status": "pass",
    }
    assert validate_second_review(json.dumps(result), "TriggerX", LoraMode.CHARACTER)["score"] == 9
    result["first_review_status"] = "needs_revision"
    with pytest.raises(CaptionValidationError):
        validate_second_review(result, "TriggerX", LoraMode.CHARACTER)


def test_new_second_review_requires_complete_chinese_translations():
    result = {
        "schema_version": 1,
        "score": 9,
        "comment": "ok",
        "final_caption": "TriggerX, 1girl, blue hair",
        "first_caption_validity": [{"tag": "1girl", "valid": True, "suggestion": None}],
        "first_review_status": "pass",
        "tag_translations": [
            {"tag": "TriggerX", "cn": "TriggerX"},
            {"tag": "1girl", "cn": "1名女孩"},
        ],
    }
    with pytest.raises(CaptionValidationError, match="blue hair"):
        validate_second_review(
            result, "TriggerX", LoraMode.CHARACTER, require_translations=True
        )
    result["tag_translations"].append({"tag": "blue hair", "cn": "蓝色头发"})
    validated = validate_second_review(
        result, "TriggerX", LoraMode.CHARACTER, require_translations=True
    )
    assert validated["tag_translations"][-1]["cn"] == "蓝色头发"


def test_second_review_allows_incomplete_translations_by_default():
    result = {
        "schema_version": 1,
        "score": 9,
        "comment": "ok",
        "final_caption": "TriggerX, 1girl, blue hair",
        "first_caption_validity": [{"tag": "1girl", "valid": True, "suggestion": None}],
        "first_review_status": "pass",
        "tag_translations": [{"tag": "TriggerX", "cn": "TriggerX"}],
    }
    assert validate_second_review(result, "TriggerX", LoraMode.CHARACTER)["score"] == 9


def test_atomic_write_never_overwrites_stage(tmp_path):
    target = tmp_path / "sample-1.txt"
    assert atomic_write(target, "one\n")
    assert is_nonempty(target)
    assert not atomic_write(target, "two\n")
    assert target.read_text() == "one\n"
    assert atomic_write(target, "three\n", overwrite=True)


def test_repository_sync_and_export(tmp_path):
    repo, asset = make_asset(tmp_path)
    first = "TriggerX, 1girl, blue hair, smile"
    repo.sync_first_tags(asset, first, "TriggerX")
    result = {
        "schema_version": 1, "score": 8, "comment": "remove smile",
        "final_caption": "TriggerX, 1girl, blue hair, looking at viewer",
        "first_caption_validity": [
            {"tag": "1girl", "valid": True, "suggestion": None},
            {"tag": "smile", "valid": False, "suggestion": "not visible"},
        ],
        "tag_translations": [
            {"tag": "TriggerX", "cn": "TriggerX"},
            {"tag": "1girl", "cn": "1名女孩"},
            {"tag": "blue hair", "cn": "蓝色头发"},
            {"tag": "smile", "cn": "微笑"},
            {"tag": "looking at viewer", "cn": "看向观众"},
        ],
        "first_review_status": "needs_revision",
    }
    repo.sync_second_tags(asset, result)
    translations = {(row["source"], row["en"]): row["cn"] for row in repo.tags(asset.image_id)}
    assert translations[("first_review", "smile")] == "微笑"
    assert translations[("second_review", "looking at viewer")] == "看向观众"
    assert all(row["cn"] for row in repo.tags(asset.image_id))
    output = ReviewService(repo).export(asset, "TriggerX", LoraMode.CHARACTER)
    text = output.read_text().strip()
    assert text.startswith("TriggerX, 1girl")
    assert "looking at viewer" in text
    assert stage_path(asset.path, 1) != output


def test_repository_ui_status_queries_and_retry(tmp_path):
    repo, asset = make_asset(tmp_path)
    run_id = "ui-status-run"
    repo._connection().execute(
        "INSERT INTO runs VALUES(?,?,?,?,?,NULL)",
        (run_id, asset.folder_id, "{}", "all", "now"),
    )
    repo.ensure_job(asset, Stage.SECOND_REVIEW, run_id, JobStatus.PENDING, 9)
    repo.set_job(
        asset.image_id,
        Stage.SECOND_REVIEW,
        JobStatus.FAILED,
        error_code="TEST",
        error_message="provider unavailable",
    )
    assert repo.stage_summary(asset.folder_id, Stage.SECOND_REVIEW) == {
        "done": 0,
        "failed": 1,
        "total": 1,
    }
    assert repo.failed_jobs(asset.folder_id)[0]["image_id"] == asset.image_id
    assert repo.image_states(asset.folder_id)[asset.image_id]["second_status"] == "failed"
    repo.retry_job(asset.image_id, Stage.SECOND_REVIEW)
    row = repo._connection().execute(
        "SELECT status,attempt_count,max_attempts FROM stage_jobs WHERE image_id=? AND stage=?",
        (asset.image_id, Stage.SECOND_REVIEW.value),
    ).fetchone()
    assert tuple(row) == ("pending", 0, 9)


def test_interrupted_running_job_is_recovered(tmp_path):
    repo, asset = make_asset(tmp_path)
    run_id = "run-test"
    repo._connection().execute("INSERT INTO runs VALUES(?,?,?,?,?,NULL)", (run_id, asset.folder_id, "{}", "all", "now"))
    repo.ensure_job(asset, Stage.FIRST_REVIEW, run_id, JobStatus.PENDING)
    repo.set_job(asset.image_id, Stage.FIRST_REVIEW, JobStatus.RUNNING)
    assert repo.recover_interrupted_jobs(asset.folder_id) == 1
    row = repo._connection().execute("SELECT status,error_code FROM stage_jobs").fetchone()
    assert (row["status"], row["error_code"]) == ("pending", "UNEXPECTED_SHUTDOWN")


def test_streaming_controller_writes_both_stages(tmp_path, monkeypatch):
    Image.new("RGB", (16, 16), "blue").save(tmp_path / "pipeline.png")

    class FakeLocal:
        def __init__(self, *args, **kwargs): pass
        def caption(self, image, prompt): return "TriggerX, 1girl, blue hair"

    class FakeCloud:
        def __init__(self, *args, **kwargs): pass
        def review(self, image, prompt, correction=""):
            return json.dumps({
                "schema_version": 1, "score": 9, "comment": "ok",
                "final_caption": "TriggerX, 1girl, blue hair",
                "first_caption_validity": [{"tag": "1girl", "valid": True, "suggestion": None}],
                "tag_translations": [
                    {"tag": "TriggerX", "cn": "TriggerX"},
                    {"tag": "1girl", "cn": "1名女孩"},
                    {"tag": "blue hair", "cn": "蓝色头发"},
                ],
                "first_review_status": "pass",
            })

    monkeypatch.setattr(controller_module, "LocalInferenceAdapter", FakeLocal)
    monkeypatch.setattr(controller_module, "CloudReviewAdapter", FakeCloud)
    monkeypatch.setattr(controller_module, "get_api_key", lambda reference: "secret-not-persisted")
    from anima_tagger.domain.models import RunConfig
    config = RunConfig(tmp_path, LoraMode.CHARACTER, "TriggerX", tmp_path / "model.gguf", tmp_path / "mmproj.gguf")
    result = controller_module.RunController(Repository(tmp_path / "pipeline.sqlite3")).run(config)
    assert result == {"first_done": 1, "second_done": 1, "failed": 0, "total": 1}
    assert (tmp_path / "pipeline-1.txt").is_file()
    assert json.loads((tmp_path / "pipeline-2.txt").read_text(encoding="utf-8"))["score"] == 9


def test_controller_does_not_retry_or_fail_for_missing_tag_translations(tmp_path, monkeypatch):
    Image.new("RGB", (16, 16), "blue").save(tmp_path / "translations.png")

    class FakeCloud:
        def __init__(self, *args, **kwargs): pass
        def review(self, image, prompt, correction=""):
            return json.dumps({
                "schema_version": 1, "score": 9, "comment": "ok",
                "final_caption": "TriggerX, 1girl, blue hair",
                "first_caption_validity": [{"tag": "1girl", "valid": True, "suggestion": None}],
                "tag_translations": [{"tag": "TriggerX", "cn": "TriggerX"}],
                "first_review_status": "pass",
            })

    monkeypatch.setattr(controller_module, "CloudReviewAdapter", FakeCloud)
    monkeypatch.setattr(controller_module, "get_api_key", lambda reference: "test-key")
    (tmp_path / "translations-1.txt").write_text("TriggerX, 1girl, blue hair\n", encoding="utf-8")
    from anima_tagger.domain.models import RunConfig
    config = RunConfig(tmp_path, LoraMode.CHARACTER, "TriggerX", tmp_path / "model.gguf", tmp_path / "mmproj.gguf")
    result = controller_module.RunController(Repository(tmp_path / "translations.sqlite3")).run(config, "second")
    assert result == {"first_done": 1, "second_done": 1, "failed": 0, "total": 1}


def test_missing_tag_translation_uses_text_only_gemma_model(tmp_path, monkeypatch):
    model = tmp_path / "model.gguf"
    model.touch()
    created = []

    class FakeLlama:
        def __init__(self, **kwargs):
            created.append(kwargs)

        def create_chat_completion(self, **kwargs):
            assert isinstance(kwargs["messages"][0]["content"], str)
            return {"choices": [{"message": {"content": '{"blue hair": "蓝色头发"}'}}]}

    monkeypatch.setitem(sys.modules, "llama_cpp", types.SimpleNamespace(Llama=FakeLlama))
    adapter = LocalInferenceAdapter(model, tmp_path / "missing-mmproj.gguf")
    assert adapter.translate_tags(["blue hair"]) == {"blue hair": "蓝色头发"}
    assert "chat_handler" not in created[0]


def test_repository_uses_beijing_time():
    assert datetime.fromisoformat(now()).utcoffset() == timedelta(hours=8)


def test_prohibited_content_is_not_retryable(monkeypatch):
    class FakeResponseError(Exception):
        status_code = 400
        body = {"message": "request blocked by Gemini API: PROHIBITED_CONTENT", "code": "prompt_blocked"}

    adapter = object.__new__(controller_module.CloudReviewAdapter)
    adapter.client = types.SimpleNamespace(
        chat=types.SimpleNamespace(
            completions=types.SimpleNamespace(create=lambda **kwargs: (_ for _ in ()).throw(FakeResponseError()))
        )
    )
    adapter.model = "test"
    image = Path(__file__)
    with pytest.raises(Exception) as exc_info:
        adapter.review(image, "prompt")
    assert exc_info.value.code == "CONTENT_BLOCKED"
    assert not exc_info.value.retryable


def test_cloud_review_uses_model_compatible_temperature():
    captured = {}

    def create(**kwargs):
        captured.update(kwargs)
        return types.SimpleNamespace(
            choices=[types.SimpleNamespace(message=types.SimpleNamespace(content="{}"))]
        )

    adapter = object.__new__(controller_module.CloudReviewAdapter)
    adapter.client = types.SimpleNamespace(
        chat=types.SimpleNamespace(completions=types.SimpleNamespace(create=create))
    )
    adapter.model = "test"

    assert adapter.review(Path(__file__), "prompt") == "{}"
    # The adapter pins a low temperature for deterministic tagging; providers that reject
    # it accept only their default of 1, so the value is asserted explicitly here.
    assert captured["temperature"] == 0.1
    request_text = captured["messages"][0]["content"][0]["text"]
    assert "comment 和 suggestion 必须使用简体中文" in request_text


def test_second_review_failure_is_recorded_in_events(tmp_path, monkeypatch):
    Image.new("RGB", (16, 16), "blue").save(tmp_path / "failed.png")

    class FakeCloud:
        def __init__(self, *args, **kwargs): pass
        def review(self, image, prompt, correction=""):
            from anima_tagger.domain.errors import CloudReviewError
            raise CloudReviewError("HTTP 400: provider rejected request", retryable=False)

    monkeypatch.setattr(controller_module, "CloudReviewAdapter", FakeCloud)
    monkeypatch.setattr(controller_module, "get_api_key", lambda reference: "test-key")
    from anima_tagger.domain.models import RunConfig
    repo = Repository(tmp_path / "failed.sqlite3")
    (tmp_path / "failed-1.txt").write_text("TriggerX, 1girl, blue hair\n", encoding="utf-8")
    config = RunConfig(tmp_path, LoraMode.CHARACTER, "TriggerX", tmp_path / "model.gguf", tmp_path / "mmproj.gguf")
    result = controller_module.RunController(repo).run(config, "second")
    events = repo.recent_events()
    assert result["failed"] == 1
    assert any(event["event_code"] == "CLOUD_REVIEW_FAILED" and "HTTP 400" in event["message"] for event in events)
