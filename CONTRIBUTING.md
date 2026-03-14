# Contributing

Thanks for your interest in contributing!

## Development setup

```bash
python -m venv .venv
./.venv/Scripts/python -m pip install -U pip
./.venv/Scripts/python -m pip install -r requirements.txt
./.venv/Scripts/python -m pip install -e .
```

## Running tests

```bash
python -m pytest -q
```

## Style guidelines

- Prefer small, focused PRs.
- Add/adjust tests for bug fixes or behavior changes.
- Keep docstrings and type hints consistent with existing code.
