import os
import json
import yaml
import logging
import hashlib
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, List


# Logging setup
def setup_logging(log_dir: Path, experiment_name: str) -> logging.Logger:
    """Initialize logger that writes to both console and file."""
    log_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = log_dir / f"{experiment_name}_{timestamp}.log"

    logger = logging.getLogger(experiment_name)
    logger.setLevel(logging.INFO)
    logger.handlers.clear()

    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )

    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)

    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(formatter)

    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    return logger


# Config management
def load_config(config_path: Path) -> Dict[str, Any]:
    """Load a YAML config file."""
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


# JSON / JSONL utilities
def save_json(data: Any, output_path: Path) -> None:
    """Save data as a pretty JSON file."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def save_jsonl(data: List[Dict], output_path: Path) -> None:
    """Save list of dicts as JSON Lines."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        for item in data:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")


def load_jsonl(input_path: Path) -> List[Dict]:
    """Load a JSONL file into a list of dicts."""
    with open(input_path, "r") as f:
        return [json.loads(line) for line in f if line.strip()]


# Experiment tracking
def get_git_commit_hash() -> str:
    """Return the current git commit hash (short)."""
    import subprocess
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            check=False,
            cwd=Path(__file__).parent.parent,
        )
        return result.stdout.strip()[:8] if result.returncode == 0 else "unknown"
    except Exception:
        return "unknown"


def create_run_id(model_name: str, strategy: str, dataset: str) -> str:
    """Generate unique run ID combining model, strategy, dataset, timestamp."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    content = f"{model_name}_{strategy}_{dataset}_{timestamp}"
    hash_suffix = hashlib.md5(content.encode()).hexdigest()[:8]
    return f"{model_name}__{strategy}__{dataset}__{hash_suffix}"


class ExperimentTracker:
    """Track runs and automatically log metadata."""

    def __init__(self, output_dir: Path):
        self.output_dir = output_dir
        self.metadata = {
            "start_time": datetime.now().isoformat(),
            "git_commit": get_git_commit_hash(),
            "runs": [],
        }

    def add_run(self, run_info: Dict[str, Any]) -> None:
        self.metadata["runs"].append(
            {**run_info, "completed_at": datetime.now().isoformat()}
        )
        self._save_metadata()

    def _save_metadata(self) -> None:
        metadata_path = self.output_dir / "experiment_metadata.json"
        with open(metadata_path, "w") as f:
            json.dump(self.metadata, f, indent=2, ensure_ascii=False)

    def get_completed_runs(self) -> List[Dict[str, Any]]:
        return self.metadata["runs"]
