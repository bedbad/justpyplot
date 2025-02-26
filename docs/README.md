# Building Documentation for justpyplot

## Requirements

Install documentation dependencies:

```bash
pip install pdoc3
```

## Building the Documentation

From the root directory of the project, run:

```bash
pdoc --html justpyplot -o docs/build --only-modules
```

This will:
1. Generate HTML documentation from docstrings
2. Output files to docs/build directory
3. Create searchable API documentation

## Development

When developing, you can use the live reload server:

```bash
pdoc --http localhost:8080 justpyplot
```

This will:
1. Start a local server
2. Auto-reload when files change
3. Show documentation updates in real-time

## Documentation Style Guide

When writing docstrings, follow this format:

```python
def function_name(param1: type, param2: type) -> return_type:
    """Short description of function.
    
    Detailed description of function behavior.
    
    Args:
        param1: Description of first parameter
        param2: Description of second parameter
        
    Returns:
        Description of return value
        
    Example:
        >>> result = function_name(1, 2)
        >>> print(result)
        3
    """
```

## Building for Distribution

For release builds:

```bash
pdoc --html justpyplot -o docs/build --only-modules --template-dir docs/templates
```

Documentation will be available at `docs/build/justpyplot/index.html`

## Read the Docs Integration

1. Go to [Read the Docs](https://readthedocs.org/) and sign up/login
2. Connect your GitHub repository
3. Import your project from the dashboard
4. The configuration in `.readthedocs.yaml` will:
   - Install required dependencies
   - Build documentation using pdoc3
   - Deploy to readthedocs.io

The documentation will automatically build when you push to the main branch.

Build Status: [![Documentation Status](https://readthedocs.org/projects/justpyplot/badge/?version=latest)](https://justpyplot.readthedocs.io/en/latest/?badge=latest)

## Troubleshooting

If builds fail:
1. Check the build logs on Read the Docs
2. Verify all dependencies are in docs/requirements.txt
3. Test the build command locally:
   ```bash
   pdoc --html justpyplot -o _readthedocs/html --only-modules
   ``` 