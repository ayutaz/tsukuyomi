# CI/CD Overview

This document describes the GitHub Actions workflows used in the Tsukuyomi TTS project.

## Workflow Structure

```
.github/workflows/
├── ci.yml          # Main CI orchestrator
├── test-full.yml   # Full test suite
├── test-quick.yml  # Quick test suite  
├── release.yml     # Release automation
└── dependencies.yml # Dependency validation
```

## Workflow Details

### 1. CI (`ci.yml`)
- **Name**: CI
- **Purpose**: Main orchestrator that routes to appropriate test suite
- **Triggers**: Push, Pull Request, Manual
- **Actions**: 
  - Runs quick tests for feature branches
  - Runs full tests for main branch and PRs

### 2. Test Suite - Full (`test-full.yml`)
- **Name**: Test Suite (Full)
- **Purpose**: Comprehensive testing across multiple environments
- **Triggers**: Called by CI, Push to main, Weekly schedule
- **Test Matrix**:
  - OS: Ubuntu 22.04, Windows, macOS
  - Python: 3.11, 3.12
- **Tests**:
  - Code quality (ruff, black, mypy)
  - Unit tests with coverage
  - Integration tests
  - Security scans
  - Documentation build
  - Package build validation

### 3. Test Suite - Quick (`test-quick.yml`)
- **Name**: Test Suite (Quick)
- **Purpose**: Fast feedback for developers
- **Triggers**: Push to dev/feature branches, PRs to dev
- **Tests**:
  - Linting and formatting
  - Basic unit tests
  - Import checks

### 4. Release (`release.yml`)
- **Name**: Release
- **Purpose**: Automated release process
- **Triggers**: Push tags (v*), Manual
- **Actions**:
  - Build Python packages
  - Create GitHub release
  - Publish to PyPI
  - Build and push Docker images

### 5. Dependencies (`dependencies.yml`)
- **Name**: Dependencies
- **Purpose**: Ensure all dependencies install correctly
- **Triggers**: 
  - Changes to requirements files
  - Weekly schedule
  - Manual
- **Tests**:
  - Python 3.11 compatibility
  - librosa dependency fix validation
  - All package imports
  - Installation script test

## Adding Status Badges

Add these badges to your README.md:

```markdown
[![CI](https://github.com/ayutaz/tsukuyomi/actions/workflows/ci.yml/badge.svg)](https://github.com/ayutaz/tsukuyomi/actions/workflows/ci.yml)
[![Dependencies](https://github.com/ayutaz/tsukuyomi/actions/workflows/dependencies.yml/badge.svg)](https://github.com/ayutaz/tsukuyomi/actions/workflows/dependencies.yml)
```

## Development Workflow

1. **Feature Development**:
   - Push to feature branch → Quick Check runs
   - Create PR → Full CI runs

2. **Main Branch**:
   - Merge to main → Full CI runs
   - Tag release → Deploy workflow runs

3. **Dependency Updates**:
   - Update requirements.txt → Dependency Tests run
   - Weekly scheduled test ensures ongoing compatibility

## Best Practices

1. **Keep workflows focused**: Each workflow has a specific purpose
2. **Use workflow calls**: Avoid duplication by calling shared workflows
3. **Matrix testing**: Test across multiple OS/Python versions in full-ci
4. **Fast feedback**: Quick checks for rapid development iteration
5. **Scheduled tests**: Weekly dependency tests catch upstream breaking changes