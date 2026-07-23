from __future__ import annotations

import os

import uvicorn

from .app import create_app

app = create_app()


def run() -> None:
    uvicorn.run(
        "airgpt_service.main:app",
        host=os.getenv("AIRGPT_HOST", "0.0.0.0"),
        port=int(os.getenv("AIRGPT_PORT", "8000")),
        reload=False,
    )


if __name__ == "__main__":
    run()
