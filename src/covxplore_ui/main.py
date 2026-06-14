from __future__ import annotations

import uvicorn


def run() -> None:
    uvicorn.run(
        "covxplore_ui.app:app",
        host="127.0.0.1",
        port=8765,
        reload=False,
    )


if __name__ == "__main__":
    run()
