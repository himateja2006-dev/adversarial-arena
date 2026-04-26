from __future__ import annotations

import uvicorn
from fastapi import FastAPI

from inference import run_episode

app = FastAPI(title="Adversarial Arena API", version="1.0.0")


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


@app.post("/reset")
def reset(task: str = "static_opponent", agent: str = "trained", episode: int = 1) -> dict:
    score = run_episode(task_name=task, agent_name=agent, episode_idx=episode, emit_logs=False)
    return {"task": task, "agent": agent, "episode": episode, "score": score}


def run_server(host: str = "0.0.0.0", port: int = 7860) -> None:
    uvicorn.run("app:app", host=host, port=port, reload=False)


if __name__ == "__main__":
    run_server()
