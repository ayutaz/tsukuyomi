"""
Tsukuyomi TTS - Setup Configuration
"""

from setuptools import setup, find_packages

setup(
    name="tsukuyomi",
    version="0.1.0",
    author="Tsukuyomi Development Team",
    description="State-of-the-art Japanese Text-to-Speech system with multilingual support",
    url="https://github.com/ayutaz/tsukuyomi",
    packages=find_packages(),
    python_requires=">=3.11",
    install_requires=[
        # Core dependencies only
        "torch>=2.0.0",
        "numpy>=1.21.0",
        "transformers>=4.30.0",
    ],
    extras_require={
        "test": [
            "pytest>=7.0.0",
            "pytest-cov>=4.0.0",
        ],
    },
)