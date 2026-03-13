"""Platform detection and environment adaptation for cross-architecture support.

Detects the current system architecture and applies necessary dependency
adjustments to oracle environment YAML configurations. This allows the same
YAML files (written for Linux x86_64) to work on ARM macOS and other platforms.
"""

import platform
import subprocess
import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any
from copy import deepcopy

logger = logging.getLogger(__name__)


@dataclass
class PlatformInfo:
    """Detected platform characteristics."""
    system: str        # 'Darwin', 'Linux', 'Windows'
    machine: str       # 'arm64', 'x86_64', 'aarch64'
    is_arm: bool = False
    is_macos: bool = False
    is_linux: bool = False
    has_cuda: bool = False

    @property
    def key(self) -> str:
        """Return a platform key for adaptation lookup (e.g. 'macos_arm64')."""
        os_part = "macos" if self.is_macos else "linux" if self.is_linux else self.system.lower()
        arch_part = "arm64" if self.is_arm else self.machine
        return f"{os_part}_{arch_part}"


def detect_platform() -> PlatformInfo:
    """Detect the current platform and its capabilities."""
    system = platform.system()
    machine = platform.machine()

    info = PlatformInfo(
        system=system,
        machine=machine,
        is_arm=machine in ('arm64', 'aarch64'),
        is_macos=system == 'Darwin',
        is_linux=system == 'Linux',
    )

    # Check for CUDA availability
    if info.is_linux:
        try:
            result = subprocess.run(
                ['nvidia-smi'], capture_output=True, text=True, timeout=5
            )
            info.has_cuda = result.returncode == 0
        except (FileNotFoundError, subprocess.TimeoutExpired):
            info.has_cuda = False

    logger.info(
        f"Detected platform: {info.system} {info.machine} "
        f"(key={info.key}, cuda={info.has_cuda})"
    )
    return info


@dataclass
class PostInstallStep:
    """A pip install to run after the conda environment is created."""
    packages: List[str]
    flags: List[str] = field(default_factory=list)
    description: str = ""


@dataclass
class EnvironmentAdaptation:
    """Describes how to adapt a YAML environment for a specific platform."""

    # Conda-level changes
    conda_remove: List[str] = field(default_factory=list)
    conda_add: List[str] = field(default_factory=list)

    # Pip-level changes (matched by package name prefix before ==, >=, etc.)
    pip_replace: Dict[str, Optional[str]] = field(default_factory=dict)
    pip_add: List[str] = field(default_factory=list)
    pip_remove: List[str] = field(default_factory=list)

    # Channels to add/remove
    channels_remove: List[str] = field(default_factory=list)

    # Post-install pip steps (run after conda env create)
    post_install: List[PostInstallStep] = field(default_factory=list)

    # Human-readable notes logged during setup
    notes: List[str] = field(default_factory=list)


def _pkg_name(spec: str) -> str:
    """Extract the package name from a conda/pip spec string.

    Examples:
        'tensorflow==2.8.0' -> 'tensorflow'
        'numpy>=1.21.0' -> 'numpy'
        'protobuf==3.20' -> 'protobuf'
    """
    # Check multi-char separators first, then single-char
    for sep in ('==', '>=', '<=', '!=', '~=', '<', '>', '='):
        if sep in spec:
            return spec.split(sep)[0].strip()
    return spec.strip()


# ---------------------------------------------------------------------------
# Platform adaptation rules per oracle
# ---------------------------------------------------------------------------
# Keys are "<oracle_name>": { "<platform_key>": EnvironmentAdaptation, ... }
# Platform keys follow PlatformInfo.key format (e.g. "macos_arm64").
#
# The YAML files in environments/ are the canonical Linux x86_64 definitions.
# Adaptations here only describe what must change for other platforms.
# ---------------------------------------------------------------------------

PLATFORM_ADAPTATIONS: Dict[str, Dict[str, EnvironmentAdaptation]] = {
    "chrombpnet": {
        "macos_arm64": EnvironmentAdaptation(
            conda_remove=["protobuf"],
            conda_add=["igraph", "leidenalg"],
            pip_replace={
                "tensorflow": "tensorflow==2.15.1",
                "tensorflow-probability": "tensorflow-probability==0.23.0",
                "tensorflow-estimator": None,  # remove, bundled in TF 2.15
                "modisco-lite": None,  # remove from pip, install post with --no-deps
            },
            pip_add=[
                "protobuf>=3.20.3,<5.0",
                "hdf5plugin",
                "numba",
            ],
            post_install=[
                PostInstallStep(
                    packages=["modisco-lite==2.0.7"],
                    flags=["--no-deps"],
                    description=(
                        "modisco-lite pins old igraph/leidenalg that require "
                        "source builds on ARM; conda-installed versions are compatible"
                    ),
                ),
            ],
            notes=[
                "TensorFlow 2.8 is not available for Apple Silicon; using 2.15.1",
                "igraph/leidenalg installed via conda to avoid source compilation",
                "modisco-lite installed with --no-deps to skip pinned igraph build",
            ],
        ),
    },
    "enformer": {
        "macos_arm64": EnvironmentAdaptation(
            pip_replace={
                "tensorflow": "tensorflow>=2.15.0,<2.17.0",
            },
            pip_add=[
                "setuptools<81",  # tensorflow_hub needs pkg_resources
            ],
            notes=[
                "TensorFlow <2.13 is not available for Apple Silicon; "
                "broadening range to >=2.15",
                "Pinned setuptools<81 (tensorflow_hub requires pkg_resources)",
            ],
        ),
    },
    "borzoi": {
        "macos_arm64": EnvironmentAdaptation(
            conda_remove=["cudatoolkit", "nvidia::cuda-nvcc"],
            channels_remove=["pytorch", "nvidia"],
            notes=[
                "CUDA packages removed (not available on macOS); "
                "PyTorch will use CPU/MPS backend",
                "Removed pytorch channel (available on conda-forge for ARM Mac)",
            ],
        ),
    },
    "sei": {
        "macos_arm64": EnvironmentAdaptation(
            conda_remove=["cudatoolkit", "pytorch", "torchvision"],
            conda_add=["pytorch>=1.13.0", "torchvision>=0.14.0"],
            channels_remove=["pytorch"],
            notes=[
                "CUDA toolkit removed (not available on macOS); "
                "PyTorch will use CPU/MPS backend",
                "Removed pytorch channel (available on conda-forge for ARM Mac)",
                "Relaxed PyTorch <2.0 upper bound (compatible with Sei on ARM Mac)",
            ],
        ),
    },
    "legnet": {
        "macos_arm64": EnvironmentAdaptation(
            conda_remove=["pytorch-gpu", "cuda-version"],
            conda_add=["pytorch>=2.0"],
            channels_remove=["pytorch", "nvidia"],
            notes=[
                "Replaced pytorch-gpu with pytorch (no GPU variant on macOS); "
                "removed CUDA, nvidia and pytorch channels (conda-forge has ARM builds)",
            ],
        ),
    },
}


def adapt_environment_config(
    config: Dict[str, Any],
    oracle: str,
    platform_info: PlatformInfo,
) -> tuple:
    """Adapt a parsed YAML environment config for the detected platform.

    Args:
        config: Parsed YAML dict (from yaml.safe_load).
        oracle: Oracle name (e.g. 'chrombpnet').
        platform_info: Detected platform information.

    Returns:
        Tuple of (adapted_config, post_install_steps, notes).
        adapted_config is a new dict (original is not mutated).
    """
    oracle_adaptations = PLATFORM_ADAPTATIONS.get(oracle, {})
    adaptation = oracle_adaptations.get(platform_info.key)

    if adaptation is None:
        return config, [], []

    config = deepcopy(config)
    deps = config.get("dependencies", [])

    # --- Channel removals ---
    if adaptation.channels_remove and "channels" in config:
        config["channels"] = [
            ch for ch in config["channels"]
            if ch not in adaptation.channels_remove
        ]

    # --- Separate conda deps and pip section ---
    conda_deps = []
    pip_section = None
    pip_section_idx = None

    for i, dep in enumerate(deps):
        if isinstance(dep, dict) and "pip" in dep:
            pip_section = dep["pip"]
            pip_section_idx = i
        else:
            conda_deps.append(dep)

    # --- Apply conda-level removals ---
    if adaptation.conda_remove:
        remove_names = set(adaptation.conda_remove)
        conda_deps = [
            d for d in conda_deps
            if _pkg_name(str(d)) not in remove_names
        ]

    # --- Apply conda-level additions ---
    for pkg in adaptation.conda_add:
        if not any(_pkg_name(str(d)) == _pkg_name(pkg) for d in conda_deps):
            conda_deps.append(pkg)

    # --- Apply pip-level changes ---
    if pip_section is not None:
        new_pip = []
        for spec in pip_section:
            name = _pkg_name(str(spec))

            # Check if this should be replaced
            if name in adaptation.pip_replace:
                replacement = adaptation.pip_replace[name]
                if replacement is not None:
                    new_pip.append(replacement)
                # else: remove (replacement is None)
                continue

            # Check if this should be removed
            if name in adaptation.pip_remove:
                continue

            new_pip.append(spec)

        # Add new pip packages
        for pkg in adaptation.pip_add:
            if not any(_pkg_name(str(p)) == _pkg_name(pkg) for p in new_pip):
                new_pip.append(pkg)

        pip_section = new_pip

    # --- Also remove pip_remove items from conda deps (e.g. 'nvidia::cuda-nvcc') ---
    if adaptation.pip_remove:
        remove_set = set(adaptation.pip_remove)
        conda_deps = [d for d in conda_deps if str(d) not in remove_set]

    # --- Rebuild dependencies list ---
    new_deps = list(conda_deps)
    if pip_section is not None:
        new_deps.append({"pip": pip_section})

    # Ensure 'pip' is in conda deps (needed for pip section)
    if pip_section is not None:
        pip_in_conda = any(str(d) == "pip" for d in new_deps if not isinstance(d, dict))
        if not pip_in_conda:
            # Insert before pip section
            new_deps.insert(len(new_deps) - 1, "pip")

    config["dependencies"] = new_deps

    return config, adaptation.post_install, adaptation.notes
