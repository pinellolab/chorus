from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

with open("requirements.txt", "r", encoding="utf-8") as fh:
    requirements = [line.strip() for line in fh if line.strip() and not line.startswith("#")]

setup(
    name="chorus",
    version="0.1.0",
    author="Pinello Lab",
    author_email="lucapinello@gmail.com",
    description="A unified interface for genomic sequence oracles",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/pinellolab/chorus",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Bio-Informatics",
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
            "pytest>=6.0",
            "pytest-cov>=2.0",
            "black>=22.0",
            "flake8>=4.0",
            "mypy>=0.950",
            "jupyter>=1.0",
            "nbconvert>=6.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "chorus=chorus.cli.main:cli",
        ],
    },
    package_data={
        "chorus": ["../environments/*.yml"],
    },
    include_package_data=True,
)