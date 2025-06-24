# CI/CD Workflow Guide

The Tsukuyomi TTS project uses GitHub Actions for comprehensive CI/CD pipelines.

## Workflow Overview

### 1. CI Orchestrator (ci.yml)
The main CI/CD orchestrator that routes to appropriate workflows based on conditions.

- **Triggers**: All pushes, pull requests, manual dispatch
- **Behavior**:
  - PR, main branch, manual dispatch → Runs Comprehensive CI
  - Other branches → Runs Quick Quality Check

### 2. Quick Quality Check (quick-check.yml)
Provides rapid feedback during development with lightweight checks.

- **Triggers**: Pushes to dev, feat/*, fix/* branches
- **Checks**:
  - Code Quality (Black, isort, Ruff, mypy)
  - Basic Functionality Tests (imports, basic unit tests)
  - File Validation & Security Scan (large files, TODOs/FIXMEs, Bandit)

### 3. Comprehensive CI (full-ci.yml)
Full quality assurance before production deployment.

- **Triggers**: main branch, pull requests, release tags
- **Checks**:
  - Multi-platform Tests (Ubuntu, Windows, macOS × Python 3.11, 3.12)
  - GPU Tests & Performance Benchmarks
  - Security & Vulnerability Scan (Trivy, Bandit)
  - Build & Package Validation
  - Docker Image Build

### 4. Release & Deploy (deploy.yml)
Manages automatic releases and package distribution.

- **Triggers**: Version tags (v*) push, manual dispatch
- **Actions**:
  - Validate Release Version
  - Build Release Artifacts
  - Create GitHub Release (with auto-generated release notes)
  - Publish to PyPI
  - Build & Push Docker Images (Docker Hub, GitHub Container Registry)

## Branch Strategy

- **main**: Production-ready, runs Comprehensive CI
- **dev**: Development branch, runs Quick Quality Check
- **feat/*, fix/**, bug/***: Feature branches, run Quick Quality Check

## Security Measures

1. **Bandit**: Python static security analysis
2. **Trivy**: Docker image and dependency vulnerability scanning
3. **Safety**: Python package known vulnerability checks
4. **pre-commit hooks**: Local development security checks

## Optimization Features

- **Concurrency Control**: Auto-cancels duplicate runs on same branch
- **Dependency Caching**: pip, apt, Docker layer caching
- **Conditional Execution**: Runs only necessary tests based on changes
- **Matrix Testing**: Parallel execution across multiple environments

## Usage

### Manual Execution
```bash
# Using GitHub CLI
gh workflow run "CI Orchestrator"
```

### Local Pre-checks
```bash
# Setup pre-commit hooks
pre-commit install

# Manual checks
make lint
make test-fast
```

### Triggering a Release
```bash
# Create and push a version tag
git tag v0.1.0
git push origin v0.1.0
```

## Troubleshooting

### Test Failures
1. Check workflow detailed logs
2. Reproduce locally with same test command
3. Run `make test-fast` for basic tests

### Build Failures
1. Verify dependencies (requirements.txt, pyproject.toml)
2. Check Python version (3.11+)
3. Look for platform-specific issues

## Future Improvements

- Kubernetes deployment automation
- Enhanced performance testing
- E2E test additions
- Load testing automation