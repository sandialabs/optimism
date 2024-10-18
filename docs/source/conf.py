# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information
import os
import sys
sys.path.insert(0, os.path.abspath('../../'))


project = 'optimism'
copyright = '2024, Brandon Talamini, Mike Tupek'
author = 'Brandon Talamini, Mike Tupek'
release = '0.1.0'

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
  'sphinx_copybutton',
  'sphinx.ext.autodoc',
  'sphinx.ext.autosummary',
  'sphinx.ext.coverage',
  'sphinx.ext.napoleon',
  'sphinx.ext.viewcode'
]

templates_path = ['_templates']
exclude_patterns = []
html_theme = 'sphinx_rtd_theme'

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_static_path = ['_static']

# autodoc_default_options = {
#   'members': True
# }
