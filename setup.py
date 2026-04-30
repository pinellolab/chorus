from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

with open("requirements.txt", "r", encoding="utf-8") as fh:
    requirements = [line.strip() for line in fh if line.strip() and not line.startswith("#")]

setup(
    name="chorus",
    version="0.4.0",
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
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
    ],
    python_requires=">=3.10",
    install_requires=requirements + ["click>=8.0"],
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
            "chorus-mcp=chorus.mcp.server:main",
        ],
    },
    package_data={
        "chorus": [
            "oracles/borzoi_source/*.txt",
            "oracles/borzoi_source/*.json",
            "oracles/sei_source/*.txt",
            "oracles/alphagenome_source/*.json",
            "analysis/data/*.bed",
            "analysis/static/*.js",  # bundled IGV.js for inline HTML reports
        ],
    },
    data_files=[
        ("chorus_environments", [
            f"environments/{f}" for f in [
                "chorus-alphagenome.yml", "chorus-borzoi.yml", "chorus-chrombpnet.yml",
                "chorus-enformer.yml", "chorus-legnet.yml", "chorus-sei.yml",
                "chorus-base.yml",
            ]
        ]),
    ],
    include_package_data=True,
)