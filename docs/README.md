# Building Documentation for justpyplot

## Requirements

Install documentation dependencies:

```bash
pip install -r docs/requirements.txt
```

The requirements include:
- sphinx
- sphinx_rtd_theme
- numpy
- opencv-python
- Pillow

## Building the Documentation

From the root directory of the project, run:

```bash
sphinx-build -b html docs docs/_build/html
```

This will:
1. Generate HTML documentation from docstrings
2. Create API reference automatically
3. Output files to docs/_build/html directory

## Development

For live preview while writing documentation:

```bash
sphinx-autobuild docs docs/_build/html
```

This will:
1. Start a local server (usually at http://127.0.0.1:8000)
2. Auto-rebuild when files change
3. Auto-reload the browser

## Documentation Style Guide

Use NumPy style docstrings for all Python functions:

```python
def function_name(param1: type, param2: type) -> return_type:
    """Short description of function.
    
    Detailed description of function behavior.
    
    Parameters
    ----------
    param1 : type
        Description of first parameter
    param2 : type
        Description of second parameter
        
    Returns
    -------
    return_type
        Description of return value
        
    Examples
    --------
    >>> result = function_name(1, 2)
    >>> print(result)
    3
    """
```

## Project Structure

```
docs/
├── conf.py          # Sphinx configuration
├── index.rst        # Main documentation page
├── requirements.txt # Documentation dependencies
├── _build/         # Generated documentation
└── _static/        # Static files (images, etc)
```

## Read the Docs Integration

The documentation automatically builds on [Read the Docs](https://readthedocs.org/) when you push to the main branch. Configuration is in `.readthedocs.yaml` at the root of the project.

## Troubleshooting

If builds fail:
1. Check the build logs on Read the Docs
2. Verify all dependencies are in docs/requirements.txt
3. Test locally with:
   ```bash
   sphinx-build -b html docs docs/_build/html -a -E
   ```
4. Clear build directory and rebuild:
   ```bash
   rm -rf docs/_build
   sphinx-build -b html docs docs/_build/html
   ``` 