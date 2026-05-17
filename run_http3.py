from __future__ import annotations

import asyncio
import os
import sys
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

from hypercorn.asyncio import serve
from hypercorn.config import Config

from main import app


BASE_DIR = Path(__file__).resolve().parent


def env_path(name: str, default: Path) -> str:
    return os.getenv(name, str(default))


async def main() -> None:
    host = os.getenv("HTTP3_HOST", "0.0.0.0")
    port = int(os.getenv("HTTP3_PORT", "8443"))
    cert_file = env_path("HTTP3_CERT_FILE", BASE_DIR / "certs" / "localhost.crt")
    key_file = env_path("HTTP3_KEY_FILE", BASE_DIR / "certs" / "localhost.key")

    config = Config()
    config.bind = [f"{host}:{port}"]
    config.quic_bind = [f"{host}:{port}"]
    config.certfile = cert_file
    config.keyfile = key_file
    config.alpn_protocols = ["h3", "h2", "http/1.1"]

    print(f"HTTP/3 server: https://localhost:{port}")
    print(f"TLS cert: {cert_file}")
    print(f"TLS key:  {key_file}")
    await serve(app, config)


if __name__ == "__main__":
    asyncio.run(main())
