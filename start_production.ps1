$ErrorActionPreference = "Stop"

$compose = Get-Command "docker" -ErrorAction SilentlyContinue
if (-not $compose) {
    throw "Docker is not installed or not on PATH."
}

docker compose up --build
