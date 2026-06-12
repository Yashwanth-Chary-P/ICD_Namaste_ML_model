"""Validate repository structure and Python syntax without writing bytecode."""

from __future__ import annotations

import ast
import importlib.util
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]

REQUIRED_PATHS = [
    "ML model/main.py",
    "ML model/accuracy.py",
    "icd-preprocessing/data/merge.py",
    "namaste-preprocessing/main.py",
    "namaste-preprocessing-validate/main.py",
    "research/error_analysis.py",
    "research/error_anlaysis_data.py",
    "research/qualitative_examples.py",
    "tm2-preprocessing/main.py",
    "requirements.txt",
    "README.md",
]

REQUIRED_PACKAGES = [
    "pandas",
    "numpy",
    "sklearn",
    "scipy",
    "tqdm",
    "rank_bm25",
]

OPTIONAL_PACKAGES = [
    "sentence_transformers",
    "torch",
]


def iter_python_files() -> list[Path]:
    ignored_parts = {".git", "__pycache__", ".venv", "venv", "env"}
    files: list[Path] = []
    for path in ROOT.rglob("*.py"):
        if ignored_parts.intersection(path.parts):
            continue
        files.append(path)
    return sorted(files)


def validate_paths() -> list[str]:
    errors: list[str] = []
    for rel_path in REQUIRED_PATHS:
        path = ROOT / rel_path
        if not path.exists():
            errors.append(f"missing required path: {rel_path}")
    return errors


def validate_python_syntax() -> list[str]:
    errors: list[str] = []
    for path in iter_python_files():
        rel_path = path.relative_to(ROOT)
        try:
            ast.parse(path.read_text(encoding="utf-8"), filename=str(rel_path))
        except SyntaxError as exc:
            errors.append(f"syntax error in {rel_path}: {exc}")
        except UnicodeDecodeError as exc:
            errors.append(f"encoding error in {rel_path}: {exc}")
    return errors


def validate_imports() -> tuple[list[str], list[str]]:
    missing_required = [
        package
        for package in REQUIRED_PACKAGES
        if importlib.util.find_spec(package) is None
    ]
    missing_optional = [
        package
        for package in OPTIONAL_PACKAGES
        if importlib.util.find_spec(package) is None
    ]
    return missing_required, missing_optional


def main() -> int:
    errors = validate_paths()
    errors.extend(validate_python_syntax())
    missing_required, missing_optional = validate_imports()

    if missing_required:
        errors.append("missing required packages: " + ", ".join(missing_required))

    if errors:
        print("Project validation failed:")
        for error in errors:
            print(f" - {error}")
        if missing_optional:
            print("Optional packages not installed: " + ", ".join(missing_optional))
        return 1

    print("Project validation passed.")
    if missing_optional:
        print("Optional packages not installed: " + ", ".join(missing_optional))
    print(f"Checked {len(iter_python_files())} Python files.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

