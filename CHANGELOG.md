# Changelog

All notable changes to Tsukuyomi TTS will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- Initial release of Tsukuyomi TTS system
- Core TTS models: XPhoneBERT, F0-BERT, VITS, Matcha-TTS, BigVGAN
- Japanese language support with G2P processing
- Multi-speaker synthesis (500+ speakers)
- Emotion control (10 emotions + VAD model)
- Style transfer with reference audio
- Voice morphing between multiple speakers
- Real-time streaming with WebSocket support
- ONNX export for Unity integration
- Docker support for Windows and Linux
- Distributed training support
- Comprehensive evaluation metrics
- API server with authentication
- LJSpeech format dataset support
- JVS dataset integration
- CLI interface

### Changed
- Optimized memory usage for consumer GPUs
- Simplified dependency management
- Improved documentation structure

### Fixed
- Import path issues
- Missing CLI entry point
- Dependency conflicts

## [0.1.0] - 2024-XX-XX (Planned)

### Features
- First stable release
- Production-ready API
- Comprehensive documentation
- Pre-trained models available