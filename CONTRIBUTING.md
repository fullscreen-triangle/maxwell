# Contributing to Maxwell

Thank you for your interest in contributing to the Maxwell project! This document provides guidelines and instructions for contributing.

## Table of Contents

- [Code of Conduct](#code-of-conduct)
- [Getting Started](#getting-started)
- [Development Setup](#development-setup)
- [Making Changes](#making-changes)
- [Testing](#testing)
- [Submitting Changes](#submitting-changes)
- [Style Guidelines](#style-guidelines)
- [Documentation](#documentation)

## Code of Conduct

This project adheres to a code of conduct. By participating, you are expected to uphold this code. Please be respectful, inclusive, and constructive in all interactions.

## Getting Started

### Prerequisites

- **Rust**: 1.75 or later (`rustup update stable`)
- **Python**: 3.10 or later (for validation)
- **Git**: For version control

### Clone the Repository

```bash
git clone https://github.com/sachikonye/maxwell.git
cd maxwell
```

## Development Setup

### Rust Processor

```bash
# Install Rust toolchain
rustup update stable
rustup component add clippy rustfmt

# Build the project
cargo build

# Run tests
cargo test

# Run the demo
cargo run -- demo
```

### Python Validation

```bash
cd validation
pip install -e ".[dev]"
python -m maxwell_validation.dissolution
```

### Full Setup

```bash
make dev  # Sets up both Rust and Python environments
```

## Making Changes

### Branching Strategy

- `main`: Stable release branch
- `develop`: Integration branch for features
- `feature/*`: Feature branches
- `fix/*`: Bug fix branches

### Creating a Branch

```bash
git checkout develop
git pull origin develop
git checkout -b feature/your-feature-name
```

### Commit Messages

Use conventional commits:

```
type(scope): description

[optional body]

[optional footer]
```

Types:
- `feat`: New feature
- `fix`: Bug fix
- `docs`: Documentation
- `style`: Formatting
- `refactor`: Code restructuring
- `test`: Adding tests
- `chore`: Maintenance

Example:
```
feat(categorical): add phase-lock network densification

Implements the network densification algorithm that increases
categorical entropy through edge addition.

Closes #42
```

## Testing

### Rust Tests

```bash
# Run all tests
cargo test

# Run specific test
cargo test test_name

# Run with output
cargo test -- --nocapture

# Run benchmarks
cargo bench
```

### Python Validation

```bash
cd validation
python -m maxwell_validation.dissolution
pytest -v
```

### Full Test Suite

```bash
make check  # Runs formatting, linting, and tests
```

## Submitting Changes

### Pull Request Process

1. Ensure all tests pass: `make check`
2. Update documentation if needed
3. Update CHANGELOG.md
4. Create a pull request against `develop`
5. Wait for review

### Pull Request Template

```markdown
## Description
[Describe your changes]

## Type of Change
- [ ] Bug fix
- [ ] New feature
- [ ] Breaking change
- [ ] Documentation update

## Checklist
- [ ] Tests pass (`cargo test`)
- [ ] Linting passes (`cargo clippy`)
- [ ] Formatting checked (`cargo fmt --check`)
- [ ] Documentation updated
- [ ] CHANGELOG.md updated
```

## Style Guidelines

### Rust

We follow the Rust style guide. Key points:

- Use `rustfmt` for formatting
- Use `clippy` for linting
- Maximum line length: 100 characters
- Use descriptive variable names
- Document public APIs with doc comments

```rust
/// Computes the categorical entropy of the phase-lock network.
///
/// # Arguments
///
/// * `network` - The phase-lock network graph
///
/// # Returns
///
/// The categorical entropy in J/K
pub fn categorical_entropy(network: &PhaseLockNetwork) -> f64 {
    // Implementation
}
```

### Python

We follow PEP 8 with these tools:

- `black` for formatting
- `ruff` for linting
- `mypy` for type checking

```python
def validate_kinetic_independence(
    positions: np.ndarray,
    velocities: np.ndarray,
) -> tuple[bool, str]:
    """
    Validate that phase-lock network is independent of velocities.
    
    Args:
        positions: Molecular positions (n, 3)
        velocities: Molecular velocities (n, 3)
    
    Returns:
        Tuple of (passed, message)
    """
    ...
```

## Documentation

### Code Documentation

- All public functions must have doc comments
- Include examples in doc comments where helpful
- Keep comments concise but informative

### LaTeX Papers

The theoretical papers are in `docs/`. When contributing to theory:

1. Follow the existing LaTeX style
2. Add references to `references.bib`
3. Update the main document if adding sections

### README Updates

Keep the README up to date with:
- New features
- Changed commands
- Updated examples

## Questions?

If you have questions, please:

1. Check existing documentation
2. Search existing issues
3. Open a new issue with the `question` label

Thank you for contributing to Maxwell! 🚀

