# GitHub Actions Workflows

This directory contains CI/CD workflows for the Tsukuyomi TTS project.

## Core Workflows

### 1. CI (`ci.yml`)
- **Trigger**: Push/PR to main/dev branches
- **Purpose**: Main CI pipeline that runs the test suite
- **Jobs**: Runs test-quick workflow

### 2. Test Suite - Quick (`test-quick.yml`)
- **Trigger**: Called by CI or manually
- **Purpose**: Fast tests for development
- **Jobs**: 
  - Code quality checks (lint, format)
  - Basic functionality tests
  - File validation & security scan

### 3. Dependencies (`dependencies.yml`)
- **Trigger**: Changes to dependency files or weekly schedule
- **Purpose**: Validate all dependencies are installable
- **Jobs**: Test installation on Python 3.11 with UV

### 4. Validate Setup (`validate-setup.yml`)
- **Trigger**: Push to main/dev or manually
- **Purpose**: Comprehensive environment validation
- **Jobs**: Full installation and import tests

### 5. Release (`release.yml`)
- **Trigger**: Push tags (v*)
- **Purpose**: Automated releases and deployment
- **Jobs**: Build, test, and publish releases

## Running Workflows Locally

To test workflows locally, use [act](https://github.com/nektos/act):

```bash
# Run CI workflow
act push -W .github/workflows/ci.yml

# Run specific job
act -j test-quick
```

## Notes

- All workflows use Python 3.11
- CPU-only PyTorch is used in CI to reduce resource usage
- Some optional dependencies may fail but won't block the workflow
- Security scans are informational only