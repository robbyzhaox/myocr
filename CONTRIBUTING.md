# Contributing to MyOCR

Thank you for your interest in contributing to MyOCR! This document provides guidelines and instructions for contributing to this project.

## Code of Conduct

By participating in this project, you agree to maintain a respectful and inclusive environment for everyone. Please be kind, considerate, and constructive in your communication.

## Getting Started

1. **Fork the repository**: Create your own fork of the repository on GitHub.
2. **Clone your fork**: 
   ```bash
   git clone https://github.com/your-username/myocr.git
   cd myocr
   ```
3. **Add the upstream repository**:
   ```bash
   git remote add upstream https://github.com/robbyzhaox/myocr.git
   ```
4. **Create a branch**: Create a new branch for your work.
   ```bash
   git checkout -b feature/your-feature-name
   ```

## Development Setup

1. **Install dependencies**:
   ```bash
   pip install -e ".[dev]"
   ```
   This installs the package in development mode with all development dependencies.

2. **Set up pre-commit hooks** (optional but recommended):
   ```bash
   pre-commit install
   ```

## Pull Request Process

1. **Keep changes focused**: Each PR should address a specific feature, bug fix, or improvement.
2. **Update documentation**: Ensure that documentation is updated to reflect your changes.
3. **Write tests**: Add or update tests for the changes you've made.
4. **Run tests locally**: Make sure all tests pass before submitting your PR.
   ```bash
   pytest
   ```
5. **Submit the PR**: Push your changes to your fork and create a PR against the main repository.
   ```bash
   git push origin feature/your-feature-name
   ```
6. **PR Description**: Provide a clear description of the changes and reference any related issues.
7. **Code Review**: Be responsive to code review comments and make necessary adjustments.

## Coding Standards

We use several tools to enforce coding standards. The easiest way to ensure your code meets these standards is by using the provided Makefile commands:

### 📦 Development Utilities

MyOCR includes several Makefile commands to help with development:

```bash
# Format code (runs isort, black, and ruff fix)
make run-format

# Run code quality checks (isort, black, ruff, mypy, pytest)
make run-checks

# Preview documentation in local
cd documentation
mkdocs serve -a 127.0.0.1:8001
```

If you prefer to run the tools individually:

**Black**: For code formatting
   ```bash
   black .
   ```

**isort**: For import sorting
   ```bash
   isort .
   ```

**Ruff**: For linting
   ```bash
   ruff check .
   ```

**mypy**: For type checking
   ```bash
   mypy myocr
   ```

The configuration for these tools is in the `pyproject.toml` file.

## Testing Guidelines

1. **Write unit tests**: Write comprehensive tests for new features and bug fixes.
2. **Test Coverage**: Aim for high test coverage for all new code.
3. **Test Directory Structure**: 
   - Place tests in the `tests/` directory
   - Follow the same directory structure as the source code

## Documentation

Good documentation is crucial for the project:

1. **Docstrings**: Add docstrings to all public classes and functions.
2. **Example Usage**: Include example usage in docstrings where appropriate.
3. **README Updates**: Update the README if you add major features or change functionality.
4. **API Documentation**: For significant additions, consider updating the API documentation.

### Building Documentation

You can build the documentation locally using:

```bash
cd documentation
mkdocs build
```

This command will generate HTML documentation and start a local server to view it.

## Issue Reporting

Before creating a new issue:

1. **Check existing issues**: Make sure the issue hasn't already been reported.
2. **Provide information**: Include detailed information about the problem:
   - Steps to reproduce
   - Expected behavior
   - Actual behavior
   - Environment (OS, Python version, etc.)
   - Logs or error messages
3. **Use templates**: If available, use the issue templates provided in the repository.

## Adding New Features

When proposing new features:

1. **Discuss first**: For major features, open an issue to discuss the feature before implementing it.
2. **Modular approach**: Keep the modular architecture in mind when designing new features.
3. **Pipeline integration**: Ensure that new components integrate well with the existing pipeline structure.
4. **Model compatibility**: If adding new models, ensure they can be loaded with the existing ModelZoo system.

## License

By contributing to MyOCR, you agree that your contributions will be licensed under the project's  [Apache 2.0 license](LICENSE).

---

Thank you for contributing to MyOCR! Your efforts help make this project better for everyone. 