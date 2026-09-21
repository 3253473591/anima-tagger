from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from anima_tagger.application.controller import RunController
from anima_tagger.domain.models import LoraMode, RunConfig, Stage
from anima_tagger.domain.validation import validate_caption
from anima_tagger.infrastructure.local_inference import LocalInferenceAdapter
from anima_tagger.infrastructure.repository import Repository
from anima_tagger.infrastructure.logging import configure_logging
from anima_tagger.prompts import load_prompt


def default_data_dir() -> Path:
    return Path(os.getenv("LOCALAPPDATA", Path.home())) / "AnimaTagger"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Anima Tagger 命令行诊断工具")
    sub = parser.add_subparsers(dest="command", required=True)
    local = sub.add_parser("test-local", help="用一张图片验证 Gemma4 多模态")
    run = sub.add_parser("run", help="执行文件夹流水线")
    for command in (local, run):
        command.add_argument("--model", type=Path, required=True)
        command.add_argument("--mmproj", type=Path, required=True)
        command.add_argument("--trigger", default="anima_test")
        command.add_argument("--mode", choices=[m.value for m in LoraMode], default="character")
        command.add_argument("--n-ctx", type=int, default=8192)
        command.add_argument("--cpu", action="store_true")
    local.add_argument("--image", type=Path, required=True)
    local.add_argument("--max-tokens", type=int, default=512)
    run.add_argument("--folder", type=Path, required=True)
    run.add_argument("--start", choices=["all", "resume", "first", "second"], default="all")
    run.add_argument("--cloud-model", default="gemini-3.5-flash")
    run.add_argument("--cloud-base-url", default="https://api.example.com/v1", help="OpenAI 兼容端点，请替换为自有的中转或官方地址")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    configure_logging(default_data_dir())
    mode = LoraMode(args.mode)
    if args.command == "test-local":
        adapter = LocalInferenceAdapter(args.model, args.mmproj, n_ctx=args.n_ctx, n_gpu_layers=0 if args.cpu else -1, verbose=False)
        result = adapter.caption(args.image, load_prompt(mode, Stage.FIRST_REVIEW, trigger_word=args.trigger), max_tokens=args.max_tokens)
        print(result)
        validate_caption(result, args.trigger, mode)
        print("MULTIMODAL_TEST_OK")
        return 0
    config = RunConfig(folder=args.folder, lora_mode=mode, trigger_word=args.trigger, gemma4_gguf_path=args.model, mmproj_path=args.mmproj,
        cloud_model=args.cloud_model, cloud_base_url=args.cloud_base_url, n_ctx=args.n_ctx, n_gpu_layers=0 if args.cpu else -1)
    repo = Repository(default_data_dir() / "anima_tagger.sqlite3")
    controller = RunController(repo, lambda stage, done, total, name: print(f"[{stage}] {done}/{total} {name}"))
    print(json.dumps(controller.run(config, args.start), ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
