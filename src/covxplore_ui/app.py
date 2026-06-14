from __future__ import annotations

import asyncio
import json
import sys
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse

from covxplore.api_client import AkaUTClient, AkaUTError
from covxplore.config import get_settings
from covxplore.prompts.registry import VARIANTS

from covxplore_ui.manager import RunManager
from covxplore_ui.schemas import RunCreateRequest, SearchRequest


manager = RunManager(default_out_dir=Path("results"))


def _configure_stdio() -> None:
    for stream_name in ("stdout", "stderr"):
        stream = getattr(sys, stream_name, None)
        if stream is None or not hasattr(stream, "reconfigure"):
            continue
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass


_configure_stdio()


def create_app() -> FastAPI:
    app = FastAPI(title="Covxplore UI API", version="0.1.0")
    app.add_middleware(
        CORSMiddleware,
        allow_origins=[
            "http://127.0.0.1:5173",
            "http://localhost:5173",
        ],
        allow_credentials=False,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    @app.get("/api/health")
    def health() -> dict[str, Any]:
        cfg = get_settings()
        akaut = {
            "baseUrl": cfg.akaut_base_url,
            "reachable": False,
            "environmentLoaded": False,
            "functionCountSample": 0,
            "error": None,
        }
        try:
            with AkaUTClient() as client:
                functions = client.search_nodes("", ["FUNCTION"])
            akaut["reachable"] = True
            akaut["environmentLoaded"] = len(functions) > 0
            akaut["functionCountSample"] = len(functions)
        except Exception as exc:
            akaut["error"] = str(exc)

        return {
            "ok": True,
            "akaut": akaut,
            "llm": {
                "model": cfg.deepseek_model,
                "baseUrl": _redact_url(cfg.deepseek_base_url),
                "hasApiKey": bool(cfg.deepseek_api_key),
            },
            "variants": sorted(VARIANTS.keys()),
        }

    @app.post("/api/akaut/search")
    def search_nodes(request: SearchRequest) -> list[dict[str, Any]]:
        try:
            with AkaUTClient() as client:
                nodes = client.search_nodes(request.query, request.types)
        except AkaUTError as exc:
            raise HTTPException(status_code=502, detail=str(exc)) from exc
        return [
            {
                "name": node.name,
                "qualifiedName": node.qualified_name,
                "absolutePath": node.absolute_path,
                "type": node.type,
                "line": node.line,
            }
            for node in nodes
        ]

    @app.get("/api/akaut/function")
    def get_function(absolutePath: str = Query(...)) -> dict[str, Any]:
        try:
            with AkaUTClient() as client:
                source = client.get_node_source(absolutePath)
                context = client.get_function_context(absolutePath)
                conditions = client.get_node_conditions(absolutePath)
        except AkaUTError as exc:
            raise HTTPException(status_code=502, detail=str(exc)) from exc
        return {
            "absolutePath": absolutePath,
            "source": source.source,
            "sourceValue": source.value,
            "context": context.context,
            "conditions": {
                "absolutePath": conditions.absolute_path,
                "totalConditions": conditions.total_conditions,
                "totalMcdcPairs": conditions.total_mcdc_pairs,
                "items": [
                    {
                        "nodeId": item.node_id,
                        "condition": item.condition,
                        "lineInFunction": item.line_in_function,
                        "startOffset": item.start_offset,
                        "endOffset": item.end_offset,
                    }
                    for item in conditions.conditions
                ],
            },
        }

    @app.post("/api/runs")
    def create_run(request: RunCreateRequest) -> dict[str, Any]:
        if request.variant not in VARIANTS:
            raise HTTPException(status_code=400, detail=f"Unknown variant: {request.variant}")
        try:
            return manager.start_run(
                absolute_path=request.absolutePath,
                variant=request.variant,
                max_iterations=request.maxIterations,
                max_tests=request.maxTests,
                mcdc_target=request.mcdcTarget,
                out_dir=request.outDir,
            )
        except RuntimeError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc

    @app.get("/api/runs")
    def list_runs() -> list[dict[str, Any]]:
        return manager.list_runs()

    @app.get("/api/runs/{run_id}")
    def get_run(run_id: str) -> dict[str, Any]:
        run = manager.get_run(run_id)
        if run is None:
            raise HTTPException(status_code=404, detail="Run not found")
        return run

    @app.get("/api/runs/{run_id}/report")
    def get_run_report(run_id: str) -> dict[str, Any]:
        report = manager.get_report(run_id)
        if report is None:
            raise HTTPException(status_code=404, detail="Run report not found")
        return report

    @app.post("/api/runs/{run_id}/cancel")
    def cancel_run(run_id: str) -> dict[str, Any]:
        run = manager.cancel(run_id)
        if run is None:
            raise HTTPException(status_code=404, detail="Run not found")
        return run

    @app.get("/api/runs/{run_id}/events")
    async def run_events(run_id: str) -> StreamingResponse:
        if manager.get_run(run_id) is None:
            raise HTTPException(status_code=404, detail="Run not found")

        async def generate():
            last_index = -1
            idle_after_done = 0
            while True:
                events, done = manager.events_since(run_id, last_index)
                for event in events:
                    last_index = max(last_index, event["index"])
                    yield _sse(event)
                if done:
                    idle_after_done += 1
                    if idle_after_done > 2:
                        break
                await asyncio.sleep(0.5)

        return StreamingResponse(generate(), media_type="text/event-stream")

    @app.get("/api/results/{run_id}")
    def get_result(run_id: str) -> dict[str, Any]:
        result = manager.get_result(run_id)
        if result is None:
            raise HTTPException(status_code=404, detail="Result not found")
        return result

    @app.get("/api/results/{run_id}/report")
    def get_result_report(run_id: str) -> dict[str, Any]:
        report = manager.get_report(run_id)
        if report is None:
            raise HTTPException(status_code=404, detail="Result report not found")
        return report

    return app


app = create_app()


def _sse(event: dict[str, Any]) -> str:
    return (
        f"event: {event['type']}\n"
        f"data: {json.dumps(event, ensure_ascii=False)}\n\n"
    )


def _redact_url(url: str) -> str:
    if "@" not in url:
        return url
    scheme, rest = url.split("://", 1) if "://" in url else ("", url)
    suffix = rest.split("@", 1)[1]
    return f"{scheme}://***@{suffix}" if scheme else f"***@{suffix}"
