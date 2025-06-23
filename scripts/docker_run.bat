@echo off
REM Docker run script for Tsukuyomi TTS (Windows)

setlocal enabledelayedexpansion

REM Check if Docker is installed
where docker >nul 2>nul
if %errorlevel% neq 0 (
    echo [ERROR] Docker is not installed. Please install Docker Desktop for Windows.
    exit /b 1
)

REM Check if docker-compose is installed
where docker-compose >nul 2>nul
if %errorlevel% neq 0 (
    echo [ERROR] docker-compose is not installed. Please install docker-compose.
    exit /b 1
)

REM Parse command line arguments
set COMMAND=%1
if "%COMMAND%"=="" set COMMAND=help
set PROFILE=%2
if "%PROFILE%"=="" set PROFILE=default

REM Set docker-compose file based on Windows version
set COMPOSE_FILE=docker-compose.windows.yml

REM Check if running on Windows with WSL2
wsl --status >nul 2>nul
if %errorlevel% equ 0 (
    echo [INFO] WSL2 detected. Using Linux containers.
    set COMPOSE_FILE=docker-compose.yml
)

if "%COMMAND%"=="build" (
    echo [INFO] Building Tsukuyomi TTS Docker image...
    docker-compose -f %COMPOSE_FILE% build
    goto :end
)

if "%COMMAND%"=="run" (
    echo [INFO] Starting Tsukuyomi TTS server...
    docker-compose -f %COMPOSE_FILE% up -d
    echo [INFO] Server started. Access:
    echo   - API: http://localhost:8000
    echo   - Demo UI: http://localhost:7860
    echo   - Docs: http://localhost:8000/docs
    goto :end
)

if "%COMMAND%"=="train" (
    echo [INFO] Starting training container...
    docker-compose -f %COMPOSE_FILE% --profile training up tsukuyomi-train
    goto :end
)

if "%COMMAND%"=="dev" (
    echo [INFO] Starting development environment...
    docker-compose -f %COMPOSE_FILE% --profile dev up -d
    echo [INFO] Jupyter notebook available at: http://localhost:8888
    goto :end
)

if "%COMMAND%"=="logs" (
    docker-compose -f %COMPOSE_FILE% logs -f tsukuyomi-tts
    goto :end
)

if "%COMMAND%"=="stop" (
    echo [INFO] Stopping Tsukuyomi TTS...
    docker-compose -f %COMPOSE_FILE% down
    goto :end
)

if "%COMMAND%"=="clean" (
    echo [WARNING] Cleaning up Docker resources...
    docker-compose -f %COMPOSE_FILE% down -v
    docker system prune -f
    goto :end
)

if "%COMMAND%"=="shell" (
    echo [INFO] Opening shell in container...
    docker-compose -f %COMPOSE_FILE% exec tsukuyomi-tts cmd
    goto :end
)

if "%COMMAND%"=="test" (
    echo [INFO] Running tests in container...
    docker-compose -f %COMPOSE_FILE% run --rm tsukuyomi-tts python -m pytest tests/ -v
    goto :end
)

if "%COMMAND%"=="benchmark" (
    echo [INFO] Running benchmarks...
    docker-compose -f %COMPOSE_FILE% run --rm tsukuyomi-tts python scripts/benchmark.py
    goto :end
)

if "%COMMAND%"=="help" (
    echo Tsukuyomi TTS Docker Helper for Windows
    echo.
    echo Usage: %0 [command] [profile]
    echo.
    echo Commands:
    echo   build      - Build Docker images
    echo   run        - Start TTS server
    echo   train      - Start training
    echo   dev        - Start development environment with Jupyter
    echo   logs       - Show container logs
    echo   stop       - Stop all containers
    echo   clean      - Clean up Docker resources
    echo   shell      - Open shell in container
    echo   test       - Run tests
    echo   benchmark  - Run benchmarks
    echo.
    echo Profiles:
    echo   default    - Basic inference server
    echo   dev        - Development with Jupyter
    echo   training   - Training environment
    echo   production - Production with Redis cache
    goto :end
)

echo [ERROR] Unknown command: %COMMAND%
echo Run '%0 help' for usage information
exit /b 1

:end
endlocal