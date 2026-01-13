# Contributing to AcouSpike

We welcome contributions to AcouSpike! This guide will help you set up your development environment and ensure your code meets our project standards.

## Development Setup

We use `uv` for dependency management.

1.  **Install dependencies** including development tools (like `ruff`):

    ```bash
    uv sync
    ```

    Alternatively, if you are not using `uv` to manage the environment, you can install `ruff` manually:

    ```bash
    pip install ruff
    ```

## Code Style & Formatting

We use **Ruff** for code formatting and linting. Before submitting your changes, please run the following commands to ensure your code is properly formatted:

1.  **Format the code:**

    ```bash
    ruff format .
    ```

2.  **Run linter (optional but recommended):**

    ```bash
    ruff check . --fix
    ```

## Contributing Workflow

1.  Fork the repository.
2.  Create a new branch for your feature or fix.
3.  Make your changes.
4.  Run `ruff format .` to clean up your code.
5.  Submit a Pull Request.

Thank you for helping improve AcouSpike!
