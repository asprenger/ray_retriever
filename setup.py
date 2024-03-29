from typing import List
import os
import re
from pathlib import Path
from setuptools import setup, find_packages

ROOT_DIR = os.path.dirname(__file__)

def read_readme() -> str:
    """Read the README file."""
    return (Path(__file__).parent / "README.md").read_text(encoding="UTF-8")

def get_path(*filepath) -> str:
    return os.path.join(ROOT_DIR, *filepath)

def find_version(filepath: str) -> str:
    """Extract version information from the given filepath.

    Adapted from https://github.com/ray-project/ray/blob/0b190ee1160eeca9796bc091e07eaebf4c85b511/python/setup.py
    """
    with open(filepath) as fp:
        version_match = re.search(r"^__version__ = ['\"]([^'\"]*)['\"]",
                                  fp.read(), re.M)
        if version_match:
            return version_match.group(1)
        raise RuntimeError("Unable to find version string.")

def get_version() -> str:
    version = find_version(get_path("ray_retriever", "__init__.py"))
    return version

requirements = [
    "ray==2.9.2",
    "ray[serve]==2.9.2",
    "langchain==0.0.340",
    "llama-index==0.9.16",
    "protobuf==3.20.3",
    "openai",
    "openapi-schema-pydantic",
    "datasets==2.15.0",
    "weaviate-client==3.23.1",
    "multiprocess==0.70.15",
    "transformers==4.33.1",
    "sentence-transformers==2.2.2",
    "typer==0.9.0",
    "rich==13.3.0",
    "torch==2.1.1",
    "python-aiconfig==1.1.24",
    "langfuse==2.12.0"
]

setup(
    name="ray_retriever",
    version="0.1.0",
    description="A scalable retrieval augmented search service.",
    author="Andre Sprenger",
    license="Apache 2.0",
    license_files="LICENSE.txt",
    long_description=read_readme(),
    long_description_content_type="text/markdown",
    url="https://github.com/asprenger/ray_retriever",
    keywords=["python", "vllm", "ray", "llm"],
    platforms= ["linux"],
    classifiers=[
        "Environment :: GPU :: NVIDIA CUDA :: 11.8",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "License :: OSI Approved :: Apache Software License",
        "Topic :: Scientific/Engineering :: Artificial Intelligence"
    ],
    packages=find_packages(include=['ray_retriever', 'ray_retriever.*']),
    install_requires=requirements,
    include_package_data=True,
    package_data={"ray_retriever": ["serve/aiconfig.yaml"]},
    python_requires=">=3.10.0",
)