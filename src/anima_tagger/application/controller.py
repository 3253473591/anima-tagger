from __future__ import annotations

import json
import logging
import random
import threading
import time
from concurrent.futures import Future, ThreadPoolExecutor, wait, FIRST_COMPLETED
from pathlib import Path
from typing import Callable

from anima_tagger.domain.errors import AnimaTaggerError, CaptionValidationError, CloudReviewError
from anima_tagger.domain.models import ImageAsset, JobStatus, RunConfig, Stage
from anima_tagger.domain.validation import normalize_first_caption, validate_caption, validate_first_caption, validate_second_review
from anima_tagger.infrastructure.cloud_review import CloudReviewAdapter
from anima_tagger.infrastructure.credentials import get_api_key
from anima_tagger.infrastructure.files import atomic_write, ensure_unchanged, is_nonempty, scan_images, stage_path
from anima_tagger.infrastructure.local_inference import LocalInferenceAdapter
from anima_tagger.infrastructure.repository import Repository
from anima_tagger.prompts import load_prompt

ProgressCallback = Callable[[str, int, int, str], None]
LogCallback = Callable[[str], None]


class RunController:
    def __init__(self, repository: Repository, progress: ProgressCallback | None = None, log: LogCallback | None = None):
        self.repository = repository
        self.progress = progress or (lambda *_: None)
        self.log = log or (lambda _: None)
        self.first_gate = threading.Event()
        self.second_gate = threading.Event()
        self.first_gate.set()
        self.second_gate.set()
        self.stop_requested = threading.Event()
        self.assets: list[ImageAsset] = []
        self._run_lock = threading.Lock()
        self._local_adapter_lock = threading.Lock()
        self._warm_local: LocalInferenceAdapter | None = None

    def set_warm_local(self, adapter: LocalInferenceAdapter) -> None:
        """Keep a UI-preloaded multimodal model for the next matching first-review run."""
        with self._local_adapter_lock:
            self._warm_local = adapter

    def _local_for(self, config: RunConfig) -> LocalInferenceAdapter:
        with self._local_adapter_lock:
            adapter = self._warm_local
        if adapter and (
            adapter.model_path == config.gemma4_gguf_path
            and adapter.mmproj_path == config.mmproj_path
            and adapter.n_ctx == config.n_ctx
            and adapter.n_gpu_layers == config.n_gpu_layers
        ):
            return adapter
        return LocalInferenceAdapter(
            config.gemma4_gguf_path,
            config.mmproj_path,
            n_ctx=config.n_ctx,
            n_gpu_layers=config.n_gpu_layers,
        )

    def _event(self, code: str, message: str, *, run_id: str | None = None, image_id: str | None = None, stage: Stage | None = None, level: str = "INFO") -> None:
        self.repository.event(code, message, run_id=run_id, image_id=image_id, stage=stage, level=level)
        line = f"{level} {code}" + (f" [{stage.value}]" if stage else "") + f": {message}"
        getattr(logging.getLogger("anima_tagger"), level.lower(), logging.getLogger("anima_tagger").info)(line)
        self.log(line)

    def pause_first(self) -> None:
        self.first_gate.clear()

    def pause_second(self) -> None:
        self.second_gate.clear()

    def resume_first(self) -> None:
        self.first_gate.set()

    def resume_second(self) -> None:
        self.second_gate.set()

    def pause_all(self) -> None:
        self.pause_first(); self.pause_second()

    def resume_all(self) -> None:
        self.resume_first(); self.resume_second()

    def _wait_gate(self, gate: threading.Event) -> None:
        while not gate.wait(0.2):
            if self.stop_requested.is_set():
                raise RuntimeError("运行已停止")

    def prepare(self, config: RunConfig, start_mode: str) -> tuple[str, list[ImageAsset]]:
        folder_id = self.repository.register_folder(config.folder)
        recovered = self.repository.recover_interrupted_jobs(folder_id) if config.resume_enabled else 0
        self.assets = scan_images(config.folder, folder_id)
        self.repository.upsert_images(self.assets)
        run_id = self.repository.create_run(config, start_mode)
        self._event("SCAN_COMPLETE", f"扫描到 {len(self.assets)} 张图片", run_id=run_id)
        if recovered:
            self._event("UNEXPECTED_SHUTDOWN_RECOVERED", f"恢复 {recovered} 个遗留任务", run_id=run_id, level="WARNING")
        return run_id, self.assets

    def run(self, config: RunConfig, start_mode: str = "all") -> dict[str, int]:
        if not self._run_lock.acquire(blocking=False):
            raise RuntimeError("当前已有运行中的任务，请使用暂停或恢复按钮")
        try:
            return self._run_locked(config, start_mode)
        finally:
            self._run_lock.release()

    def _run_locked(self, config: RunConfig, start_mode: str) -> dict[str, int]:
        if start_mode not in {"all", "resume", "first", "second"}:
            raise ValueError("未知启动模式")
        run_id, assets = self.prepare(config, start_mode)
        local = None
        if start_mode != "second":
            local = self._local_for(config)
        cloud = None
        api_key = get_api_key(config.gemini_api_key_ref)
        if api_key:
            cloud = CloudReviewAdapter(api_key, config.cloud_model, config.cloud_base_url, config.proxy_url)

        first_needed: list[ImageAsset] = []
        second_ready: list[ImageAsset] = []
        second_seen: set[str] = set()
        counts = {"first_done": 0, "second_done": 0, "failed": 0, "total": len(assets)}
        for asset in assets:
            first_file, second_file = stage_path(asset.path, 1), stage_path(asset.path, 2)
            if is_nonempty(first_file):
                self.repository.ensure_job(asset, Stage.FIRST_REVIEW, run_id, JobStatus.SKIPPED)
                try:
                    caption = first_file.read_text(encoding="utf-8").strip()
                    validate_first_caption(caption, config.trigger_word, config.lora_mode)
                    self.repository.sync_first_tags(asset, caption, config.trigger_word)
                except Exception as exc:
                    self._event("EXISTING_FIRST_INVALID", str(exc), run_id=run_id, image_id=asset.image_id, stage=Stage.FIRST_REVIEW, level="WARNING")
                counts["first_done"] += 1
            elif start_mode != "second":
                self.repository.ensure_job(asset, Stage.FIRST_REVIEW, run_id, JobStatus.PENDING)
                first_needed.append(asset)
            if is_nonempty(second_file):
                self.repository.ensure_job(
                    asset, Stage.SECOND_REVIEW, run_id, JobStatus.SKIPPED, config.api_retry_count
                )
                self._sync_existing_second(asset, config)
                counts["second_done"] += 1
                second_seen.add(asset.image_id)
            elif is_nonempty(first_file):
                self.repository.ensure_job(
                    asset, Stage.SECOND_REVIEW, run_id, JobStatus.PENDING, config.api_retry_count
                )
                second_ready.append(asset); second_seen.add(asset.image_id)

        # llama.cpp owns one shared model instance; unsafe requested concurrency is capped.
        first_workers = 1
        if config.first_review_concurrency > 1:
            self._event("LOCAL_CONCURRENCY_CAPPED", "共享 Gemma4 实例的实际并发已降为 1", run_id=run_id, stage=Stage.FIRST_REVIEW, level="WARNING")
        with ThreadPoolExecutor(max_workers=first_workers, thread_name_prefix="first") as first_pool, ThreadPoolExecutor(max_workers=config.second_review_concurrency, thread_name_prefix="second") as second_pool:
            first_futures: dict[Future, ImageAsset] = {}
            second_futures: dict[Future, ImageAsset] = {}
            first_index = 0
            second_index = 0
            while first_index < len(first_needed) or first_futures or second_index < len(second_ready) or second_futures:
                while self.first_gate.is_set() and first_index < len(first_needed) and len(first_futures) < first_workers:
                    asset = first_needed[first_index]; first_index += 1
                    first_futures[first_pool.submit(self._first, asset, config, local, run_id)] = asset
                while cloud and self.second_gate.is_set() and second_index < len(second_ready) and len(second_futures) < config.second_review_concurrency:
                    asset = second_ready[second_index]; second_index += 1
                    second_futures[second_pool.submit(self._second, asset, config, cloud, run_id)] = asset
                active = set(first_futures) | set(second_futures)
                if not active:
                    if not cloud and second_index < len(second_ready):
                        for asset in second_ready[second_index:]:
                            self.repository.set_job(asset.image_id, Stage.SECOND_REVIEW, JobStatus.BLOCKED, error_code="API_KEY_MISSING", error_message="未配置云端 API Key")
                            self._event("API_KEY_MISSING", f"{asset.path.name}: 未配置云端 API Key", run_id=run_id, image_id=asset.image_id, stage=Stage.SECOND_REVIEW, level="ERROR")
                        break
                    time.sleep(0.1); continue
                done, _ = wait(active, timeout=0.2, return_when=FIRST_COMPLETED)
                for future in done:
                    if future in first_futures:
                        asset = first_futures.pop(future)
                        if self._future_ok(future, counts, asset, Stage.FIRST_REVIEW, run_id):
                            counts["first_done"] += 1
                            if asset.image_id not in second_seen and not is_nonempty(stage_path(asset.path, 2)):
                                self.repository.ensure_job(
                                    asset, Stage.SECOND_REVIEW, run_id, JobStatus.PENDING, config.api_retry_count
                                )
                                second_ready.append(asset); second_seen.add(asset.image_id)
                        self.progress("first", counts["first_done"], len(assets), asset.path.name)
                    else:
                        asset = second_futures.pop(future)
                        if self._future_ok(future, counts, asset, Stage.SECOND_REVIEW, run_id): counts["second_done"] += 1
                        self.progress("second", counts["second_done"], len(assets), asset.path.name)
        self._event("RUN_FINISHED", json.dumps(counts), run_id=run_id, level="WARNING" if counts["failed"] else "INFO")
        return counts

    def _future_ok(self, future: Future, counts: dict[str, int], asset: ImageAsset, stage: Stage, run_id: str) -> bool:
        try:
            future.result(); return True
        except Exception as exc:
            counts["failed"] += 1
            self._event(getattr(exc, "code", "TASK_FAILED"), f"{asset.path.name}: {exc}", run_id=run_id, image_id=asset.image_id, stage=stage, level="ERROR")
            return False

    def _first(self, asset: ImageAsset, config: RunConfig, local: LocalInferenceAdapter, run_id: str) -> None:
        self._wait_gate(self.first_gate)
        self.repository.set_job(asset.image_id, Stage.FIRST_REVIEW, JobStatus.RUNNING, increment=True)
        try:
            ensure_unchanged(asset)
            prompt = load_prompt(config.lora_mode, Stage.FIRST_REVIEW, trigger_word=config.trigger_word)
            caption = local.caption(asset.path, prompt)
            validate_first_caption(caption, config.trigger_word, config.lora_mode)
            if config.first_caption_rule_mode == "filter":
                caption = normalize_first_caption(caption, config.trigger_word, config.lora_mode)
            ensure_unchanged(asset)
            if not atomic_write(stage_path(asset.path, 1), caption + "\n") and not is_nonempty(stage_path(asset.path, 1)):
                raise OSError("一审文件写入失败")
            self.repository.sync_first_tags(asset, caption, config.trigger_word)
            self.repository.set_job(asset.image_id, Stage.FIRST_REVIEW, JobStatus.SUCCEEDED)
        except Exception as exc:
            code = getattr(exc, "code", "LOCAL_INFERENCE_FAILED")
            self.repository.set_job(asset.image_id, Stage.FIRST_REVIEW, JobStatus.FAILED, error_code=code, error_message=str(exc))
            self._event(code, f"{asset.path.name}: {exc}", run_id=run_id, image_id=asset.image_id, stage=Stage.FIRST_REVIEW, level="ERROR")
            raise

    @staticmethod
    def _should_send_correction(exc: Exception) -> bool:
        """返回 True 表示错误信息对模型修正有帮助，应该发送。"""
        msg = str(exc)
        # JSON 解析失败 — 模型不需要知道具体位置，只要重新输出合法 JSON 即可
        if "不是有效 JSON" in msg:
            return False
        # 内容被云端拦截 — 重试也不会改变结果
        if "prompt_blocked" in msg.casefold() or "PROHIBITED_CONTENT" in msg:
            return False
        return True

    def _second(self, asset: ImageAsset, config: RunConfig, cloud: CloudReviewAdapter, run_id: str) -> None:
        first_caption = stage_path(asset.path, 1).read_text(encoding="utf-8").strip()
        prompt = load_prompt(
            config.lora_mode,
            Stage.SECOND_REVIEW,
            trigger_word=config.trigger_word,
            first_caption=first_caption,
        )
        last: Exception | None = None
        for attempt in range(1, config.api_retry_count + 1):
            self._wait_gate(self.second_gate)
            self.repository.set_job(asset.image_id, Stage.SECOND_REVIEW, JobStatus.RUNNING, increment=True)
            try:
                ensure_unchanged(asset)
                correction = ""
                if attempt > 1 and last is not None and self._should_send_correction(last):
                    correction = f"\n上次输出无效，错误原因: {last}。请根据错误原因修正输出。"
                raw = cloud.review(asset.path, prompt, correction)
                result = validate_second_review(
                    raw,
                    config.trigger_word,
                    config.lora_mode,
                )
                ensure_unchanged(asset)
                atomic_write(stage_path(asset.path, 2), json.dumps(result, ensure_ascii=False, indent=2) + "\n")
                self.repository.sync_second_tags(asset, result)
                self.repository.set_job(asset.image_id, Stage.SECOND_REVIEW, JobStatus.SUCCEEDED)
                return
            except Exception as exc:
                last = exc
                retryable = isinstance(exc, CaptionValidationError) or getattr(exc, "retryable", False)
                self._event(
                    getattr(exc, "code", "INVALID_SECOND_REVIEW_JSON"),
                    f"{asset.path.name} 第 {attempt}/{config.api_retry_count} 次失败: {exc}",
                    run_id=run_id,
                    image_id=asset.image_id,
                    stage=Stage.SECOND_REVIEW,
                    level="WARNING" if retryable and attempt < config.api_retry_count else "ERROR",
                )
                if not retryable or attempt == config.api_retry_count:
                    break
                self.repository.set_job(asset.image_id, Stage.SECOND_REVIEW, JobStatus.RETRY_WAIT, error_code=getattr(exc, "code", "INVALID_SECOND_REVIEW_JSON"), error_message=str(exc))
                time.sleep(min(2 ** (attempt - 1) + random.random(), 15))
        self.repository.set_job(asset.image_id, Stage.SECOND_REVIEW, JobStatus.FAILED, error_code=getattr(last, "code", "CLOUD_REVIEW_FAILED"), error_message=str(last))
        raise last or CloudReviewError("二审失败")

    def _sync_existing_second(self, asset: ImageAsset, config: RunConfig) -> None:
        try:
            raw = stage_path(asset.path, 2).read_text(encoding="utf-8")
            self.repository.sync_second_tags(asset, validate_second_review(raw, config.trigger_word, config.lora_mode))
        except Exception as exc:
            self._event("EXISTING_SECOND_INVALID", str(exc), image_id=asset.image_id, stage=Stage.SECOND_REVIEW, level="WARNING")
