from __future__ import annotations

from pathlib import Path
import re

from setuptools import find_packages, setup


def _read_readme() -> str:
    readme_path = Path(__file__).with_name("README.md")
    if readme_path.exists():
        return readme_path.read_text(encoding="utf-8")
    return (
        "HighDim-Inference-Toolkit: high-dimensional statistical inference utilities."
    )


def _read_version() -> str:
    version_path = Path(__file__).parent / "highdim_inference_toolkit" / "__init__.py"
    text = version_path.read_text(encoding="utf-8")
    match = re.search(r"^__version__\s*=\s*\"([^\"]+)\"\s*$", text, re.M)
    if not match:
        raise RuntimeError(
            "Could not find __version__ in highdim_inference_toolkit/__init__.py"
        )
    return match.group(1)


setup(
    name="highdim-inference-toolkit",
    version=_read_version(),
    description="High-dimensional statistical inference toolkit (Lasso, Debiased Lasso, Trans-Lasso).",
    long_description=_read_readme(),
    long_description_content_type="text/markdown",
    license="MIT",
    license_files=("LICENSE",),
    url="https://github.com/leoplasture/HighDim-Inference-Toolkit",
    project_urls={
        "Source": "https://github.com/leoplasture/HighDim-Inference-Toolkit",
        "Issues": "https://github.com/leoplasture/HighDim-Inference-Toolkit/issues",
    },
    packages=find_packages(exclude=("tests", "examples", "data", "src")),
    python_requires=">=3.9",
    install_requires=[
        "numpy>=1.23",
        "scipy>=1.10",
        "scikit-learn>=1.2",
        "matplotlib>=3.7",
    ],
    extras_require={
        "dev": [
            "pytest>=7.0",
        ]
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3 :: Only",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Mathematics",
    ],
    keywords=[
        "lasso",
        "debiased lasso",
        "high-dimensional",
        "inference",
        "transfer learning",
    ],
    include_package_data=True,
)
