import os
import sys

# Add the project root directory to Python path
sys.path.insert(0, os.path.abspath('..'))

# Configuration file for the Sphinx documentation builder
project = 'justpyplot'
copyright = '2024'
author = 'bedbad'

# Basic Sphinx settings
extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.napoleon',  # For NumPy style docstrings
    'sphinx.ext.viewcode',  # To show source code
]

# Autodoc settings
autodoc_default_options = {
    'members': None,
    'undoc-members': False,
    'private-members': False,
    'special-members': False,
    'imported-members': False
}

def skip_non_all(app, what, name, obj, skip, options):
    # Get the module name and member name
    module_name = name.rsplit('.', 1)[0]
    member_name = name.split('.')[-1]
    
    try:
        # Get the module's __all__
        module = sys.modules[module_name]
        all_list = getattr(module, '__all__', [])
        
        # Skip if not in __all__
        if member_name not in all_list:
            return True
    except (KeyError, AttributeError):
        pass
    
    return skip

def setup(app):
    app.connect('autodoc-skip-member', skip_non_all)

templates_path = ['_templates']
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']
html_theme = 'sphinx_rtd_theme'

# Napoleon settings
napoleon_google_docstring = False
napoleon_numpy_docstring = True