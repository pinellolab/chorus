from pathlib import Path

from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

with open("requirements.txt", "r", encoding="utf-8") as fh:
    requirements = [line.strip() for line in fh if line.strip() and not line.startswith("#")]

setup(
    name="chorus",
    version="0.7.2",
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
            "nbformat>=5.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "chorus=chorus.cli.main:cli",
            "chorus-mcp=chorus.mcp.server:main",
        ],
    },
    package_data={
        # Every non-.py file that shipped code READS must be listed, or a
        # non-editable install (`pip install .`, `pip install git+...`) ships a
        # package that cannot load the metadata it needs. Six were missing, each
        # read unconditionally with no fallback:
        #   cherimoya_source/CATv1-metadata.tsv          catv1_metadata.py
        #   cherimoya_source/CATv1-performance-fold0.tsv  catv1_metadata.py
        #   chrombpnet_source/chrombpnet_JASPAR_metadata.tsv  chrombpnet.py
        #   chrombpnet_source/templates/input_data.json   bpnet.py
        #   enformer_source/enformer_human_targets.txt    enformer_metadata.py
        #   sei_source/target.names                       weights_probe.py
        # The README documents the editable install, which is why this only bit
        # users who deviated -- but setup.py declares a console_scripts entry point,
        # so a normal install is a supported thing to attempt.
        # tests/test_packaging_ships_the_data_it_reads.py enumerates them from the
        # tree rather than trusting this list to stay complete.
        "chorus": [
            "oracles/borzoi_source/*.txt",
            # (no *.json here: borzoi_source has never contained one. The dead
            # pattern was harmless in itself but hid the six real gaps above,
            # because a list with entries in it looks maintained.)
            "oracles/sei_source/*.txt",
            "oracles/sei_source/target.names",
            "oracles/alphagenome_source/*.json",
            "oracles/cherimoya_source/*.tsv",
            "oracles/chrombpnet_source/*.tsv",
            "oracles/chrombpnet_source/templates/*.json",
            "oracles/enformer_source/*.txt",
            "analysis/data/*.bed",
            "analysis/static/*.js",  # bundled IGV.js for inline HTML reports
            # The hg38 cytoband table, which supplies both the ideogram and the
            # chromosome lengths a report inlines instead of fetching (#139).
            # Globbed by extension rather than named: `*.js` alone silently
            # dropped this file from the wheel, and the symptom would have been a
            # pip-installed chorus quietly falling back to igv.org's registry.
            "analysis/static/*.txt.gz",
        ],
    },
    data_files=[
        # Listed by glob rather than by hand: the hand-written list omitted
        # chorus-cherimoya, chorus-epinformerseq and chorus-alphagenome_pt, so
        # `chorus setup --oracle cherimoya` had no yml to read from a non-editable
        # install. NOTE this lands in `<prefix>/chorus_environments/` while
        # CHORUS_ENVIRONMENTS_DIR resolves to `<repo>/environments`, so nothing reads
        # the installed copy today -- see the packaging test, which records that
        # mismatch rather than leaving it implied.
        ("chorus_environments", sorted(str(p) for p in Path("environments").glob("*.yml"))),
    ],
    include_package_data=True,
)