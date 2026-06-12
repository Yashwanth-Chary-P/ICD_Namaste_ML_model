# Validation

## Repository Validation

The repository includes a lightweight validation helper:

```powershell
python scripts/validate_project.py
```

It verifies:

- Required project files and folders exist.
- Python files parse successfully using `ast`, without writing `__pycache__`.
- Required Python packages are importable in the active environment.
- Optional dense retrieval packages are reported separately when missing.

## Latest Local Validation

Validated from the project root with:

```powershell
python scripts/validate_project.py
```

Result:

```text
Project validation passed.
Checked 32 Python files.
```

