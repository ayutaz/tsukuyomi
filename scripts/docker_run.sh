#!/bin/bash
# Docker run script for Tsukuyomi TTS (Linux/macOS)

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Function to print colored output
print_info() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

# Check if Docker is installed
if ! command -v docker &> /dev/null; then
    print_error "Docker is not installed. Please install Docker first."
    exit 1
fi

# Check if docker-compose is installed
if ! command -v docker-compose &> /dev/null; then
    print_error "docker-compose is not installed. Please install docker-compose first."
    exit 1
fi

# Check for NVIDIA Docker runtime
if docker info 2>/dev/null | grep -q nvidia; then
    print_info "NVIDIA Docker runtime detected"
    export DOCKER_RUNTIME="nvidia"
else
    print_warning "NVIDIA Docker runtime not found. Running in CPU mode."
    export DOCKER_RUNTIME="runc"
fi

# Parse command line arguments
COMMAND=${1:-"help"}
PROFILE=${2:-"default"}

case $COMMAND in
    build)
        print_info "Building Tsukuyomi TTS Docker image..."
        docker-compose build
        ;;
    
    run)
        print_info "Starting Tsukuyomi TTS server..."
        docker-compose up -d
        print_info "Server started. Access:"
        print_info "  - API: http://localhost:8000"
        print_info "  - Demo UI: http://localhost:7860"
        print_info "  - Docs: http://localhost:8000/docs"
        ;;
    
    train)
        print_info "Starting training container..."
        docker-compose --profile training up tsukuyomi-train
        ;;
    
    dev)
        print_info "Starting development environment..."
        docker-compose --profile dev up -d
        print_info "Jupyter notebook available at: http://localhost:8888"
        ;;
    
    logs)
        docker-compose logs -f tsukuyomi-tts
        ;;
    
    stop)
        print_info "Stopping Tsukuyomi TTS..."
        docker-compose down
        ;;
    
    clean)
        print_warning "Cleaning up Docker resources..."
        docker-compose down -v
        docker system prune -f
        ;;
    
    shell)
        print_info "Opening shell in container..."
        docker-compose exec tsukuyomi-tts /bin/bash
        ;;
    
    test)
        print_info "Running tests in container..."
        docker-compose run --rm tsukuyomi-tts python -m pytest tests/ -v
        ;;
    
    benchmark)
        print_info "Running benchmarks..."
        docker-compose run --rm tsukuyomi-tts python scripts/benchmark.py
        ;;
    
    help)
        echo "Tsukuyomi TTS Docker Helper"
        echo ""
        echo "Usage: $0 [command] [profile]"
        echo ""
        echo "Commands:"
        echo "  build      - Build Docker images"
        echo "  run        - Start TTS server"
        echo "  train      - Start training"
        echo "  dev        - Start development environment with Jupyter"
        echo "  logs       - Show container logs"
        echo "  stop       - Stop all containers"
        echo "  clean      - Clean up Docker resources"
        echo "  shell      - Open shell in container"
        echo "  test       - Run tests"
        echo "  benchmark  - Run benchmarks"
        echo ""
        echo "Profiles:"
        echo "  default    - Basic inference server"
        echo "  dev        - Development with Jupyter"
        echo "  training   - Training environment"
        echo "  production - Production with Redis cache"
        ;;
    
    *)
        print_error "Unknown command: $COMMAND"
        echo "Run '$0 help' for usage information"
        exit 1
        ;;
esac