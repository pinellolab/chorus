# Contributing to Chorus

We welcome contributions to Chorus! This guide will help you get started.

## Getting Started

1. Fork the repository on GitHub
2. Clone your fork locally:
   ```bash
   git clone https://github.com/YOUR_USERNAME/chorus.git
   cd chorus
   ```
3. Create a new branch for your feature:
   ```bash
   git checkout -b feature/your-feature-name
   ```
4. Install the development dependencies:
   ```bash
   pip install -e ".[dev]"
   ```

## Development Guidelines

### Code Style

- We follow PEP 8 Python style guidelines
- Use meaningful variable and function names
- Add docstrings to all functions and classes
- Keep functions focused and modular

### Running Tests

Run the test suite before submitting:
```bash
pytest tests/
```

Run with coverage:
```bash
pytest --cov=chorus tests/
```

### Code Formatting

Format your code with black:
```bash
black chorus/
```

Check style with flake8:
```bash
flake8 chorus/
```

### Type Checking

Run mypy for type checking:
```bash
mypy chorus/
```

## Making Changes

1. Make your changes in your feature branch
2. Add or update tests as needed
3. Update documentation if you're changing functionality
4. Ensure all tests pass
5. Commit your changes with a clear message:
   ```bash
   git commit -m "Add feature: brief description"
   ```

## Submitting a Pull Request

1. Push your branch to your fork:
   ```bash
   git push origin feature/your-feature-name
   ```
2. Open a pull request on GitHub
3. Describe your changes in detail
4. Link any relevant issues

## Adding New Oracles

To add a new oracle model:

1. Create a new file in `chorus/oracles/`
2. Inherit from `OracleBase`
3. Implement all abstract methods:
   - `load_pretrained_model()`
   - `list_assay_types()`
   - `list_cell_types()`
   - `_predict()`
   - `fine_tune()`
   - `_get_context_size()`
   - `_get_sequence_length_bounds()`
   - `_get_bin_size()`

4. Add your oracle to `chorus/oracles/__init__.py`
5. Create tests in `tests/test_oracles.py`
6. Add an example notebook in `examples/`

## Reporting Issues

- Use GitHub Issues to report bugs
- Include a minimal reproducible example
- Specify your environment (OS, Python version, package versions)
- Describe expected vs actual behavior

## Code of Conduct

- Be respectful and inclusive
- Welcome newcomers and help them get started
- Focus on constructive criticism
- Respect differing viewpoints and experiences

## Questions?

Feel free to open an issue for any questions about contributing!