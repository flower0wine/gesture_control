# AGENTS.md

This document provides guidelines for AI coding agents working on this project.

## Project Overview

- **Language**: Python 3.12+
- **Package Manager**: uv
- **Type**: MediaPipe integration project

## Build Commands

### Package Management (uv)

```bash
# Install dependencies
uv sync

# Add a dependency
uv add <package>
uv add --dev <package>  # dev dependency

# Remove a dependency
uv remove <package>

# Update dependencies
uv update
uv pip install -e .  # editable install
```

### Running the Application

```bash
# Run main module
python -m main

# Or run directly
python main.py
```

### Type Checking

```bash
# Run mypy on entire project
uv run mypy .

# Run mypy on specific file
uv run mypy src/module.py
```

### Linting & Formatting

```bash
# Run ruff linter
uv run ruff check .

# Run ruff formatter
uv run ruff format .

# Auto-fix issues
uv run ruff check --fix .
uv run ruff format --check .  # check without modifying
```

### Testing

```bash
# Run all tests with pytest
uv run pytest

# Run a single test file
uv run pytest tests/test_module.py

# Run a single test function
uv run pytest tests/test_module.py::test_function_name

# Run tests with verbose output
uv run pytest -v

# Run tests with coverage
uv run pytest --cov=src --cov-report=term-missing
```

## Code Style Guidelines

### General Principles

- Write clean, readable, and maintainable code
- Follow PEP 8 style guide
- Prefer explicit over implicit
- "There should be one obvious way to do it" (PEP 20)

### Imports

```python
# Standard library imports first
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

# Third-party imports
import numpy as np
from PIL import Image

# Local application imports
from src.module import ClassName
from src.utils import helper_function
```

### File Structure

```python
"""
Module docstring explaining the purpose of this file.
"""

# Standard library imports
import ...

# Third-party imports
import ...

# Local imports
import ...

# Constants (UPPER_SNAKE_CASE)
DEFAULT_CONFIG = {...}

# Classes
class MyClass:
    """Class docstring."""

    CONSTANT = "value"

    def __init__(self, param: Type) -> None:
        """Initialize the class."""
        ...

    def method(self, arg: Type) -> ReturnType:
        """Method docstring."""
        ...


# Functions (snake_case)
def function_name(param: Type) -> ReturnType:
    """Function docstring."""
    ...


if __name__ == "__main__":
    main()
```

### Naming Conventions

| Element | Convention | Example |
|---------|------------|---------|
| Modules | snake_case | `data_processing` |
| Classes | PascalCase | `FaceDetector` |
| Functions | snake_case | `detect_landmarks` |
| Variables | snake_case | `image_path` |
| Constants | UPPER_SNAKE_CASE | `MAX_DIMENSION` |
| Type Variables | PascalCase | `T`, `StateT` |
| Private Attributes | _snake_case | `_internal_cache` |

### Type Hints

```python
# Always use type hints for function signatures
def process_image(path: Path, options: Optional[Dict[str, Any]] = None) -> List[np.ndarray]:
    ...

# Use | instead of Union for Python 3.10+
def func(x: int | None) -> str: ...

# Use @dataclass for simple data classes
from dataclasses import dataclass

@dataclass
class Config:
    """Configuration for the detector."""
    threshold: float = 0.5
    max_fps: int = 30

# Generic types
from typing import TypeVar, Generic

T = TypeVar("T")

class Container(Generic[T]):
    def __init__(self, value: T) -> None:
        self.value = value
```

### Error Handling

```python
# Use custom exceptions for domain-specific errors
class MediaPipeError(Exception):
    """Base exception for MediaPipe-related errors."""


class ModelLoadError(MediaPipeError):
    """Failed to load the model."""


# Handle errors at the appropriate level
def load_model(path: Path) -> Calculator:
    if not path.exists():
        raise ModelLoadError(f"Model file not found: {path}")
    try:
        return Calculator.create(str(path))
    except Exception as e:
        raise ModelLoadError(f"Failed to load model: {e}") from e


# Use context managers for resource handling
with Image.open(image_path) as img:
    process(img)


# Never use bare except clauses
try:
    ...
except ValueError as e:
    logger.error(f"Invalid value: {e}")
    raise
```

### Documentation

```python
def complex_function(
    param1: str,
    param2: int,
    param3: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """
    Brief description of what the function does.

    Args:
        param1: Description of param1.
        param2: Description of param2.
        param3: Optional list of strings. Defaults to None.

    Returns:
        Dictionary containing results with keys 'result' and 'metadata'.

    Raises:
        ValueError: When param2 is negative.
        TypeError: When param1 is not a valid string.
    """
    ...
```

### Testing

```python
import pytest

class TestCalculator:
    def test_add_two_numbers(self) -> None:
        assert add(2, 3) == 5

    def test_with_mock(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr("module.function", mock_function)
        ...
```

## Development Workflow

1. Create a feature branch for new work
2. Run linting and type checking before committing
3. Write tests for new functionality
4. Ensure all tests pass before merging
5. Use meaningful commit messages

## Key Tools

- **uv**: Fast Python package manager
- **ruff**: Linter and formatter (extremely fast)
- **mypy**: Static type checker
- **pytest**: Testing framework
