"""
Tsukuyomi TTS - Setup Configuration
"""

from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

with open("requirements.txt", "r", encoding="utf-8") as fh:
    requirements = [line.strip() for line in fh if line.strip() and not line.startswith("#")]

setup(
    name="tsukuyomi-tts",
    version="0.1.0",
    author="Tsukuyomi Development Team",
    description="State-of-the-art Japanese Text-to-Speech system with multilingual support",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/tsukuyomi",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Multimedia :: Sound/Audio :: Speech",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
    python_requires=">=3.8",
    install_requires=requirements,
    extras_require={
        "dev": [
            "pytest>=7.0.0",
            "black>=23.0.0",
            "flake8>=6.0.0",
            "isort>=5.12.0",
            "pre-commit>=3.0.0",
        ],
        "h100": [
            "transformer-engine>=1.2.0",
            "flash-attn>=2.5.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "tsukuyomi=src.cli:main",
        ],
    },
    include_package_data=True,
    package_data={
        "tsukuyomi": ["configs/*.yaml", "configs/*.json"],
    },
)