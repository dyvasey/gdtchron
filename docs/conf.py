# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = 'GDTchron'
copyright = '2025, Dylan Vasey, Peter Scully, John Naliboff'
author = 'Dylan Vasey, Peter Scully, John Naliboff'
release = '0.1.0'

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    "nbsphinx",
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",           
    "sphinx_autodoc_typehints",
    "myst_parser"
]

templates_path = ['_templates']
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']

autosummary_generate = True

autodoc_default_options = {
    "members": True,
}

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = 'sphinx_rtd_theme'
html_logo = 'media/logo.png'
html_static_path = ['_static']

def setup(app):
    app.add_css_file("custom.css")

# Path setup
import os
import sys
sys.path.insert(0, os.path.abspath('..'))

# Docs from root repository
import shutil
shutil.copy('../CONTRIBUTING.md', './CONTRIBUTING.md')
shutil.copy('../README.md', './README.md')

# Delete logo from README
with open('README.md','r') as file:
    lines = file.readlines()

with open('README.md','w') as file:
    file.writelines(lines[3:])

try:
    shutil.copytree('../media','./media')
except FileExistsError:
    shutil.rmtree('./media')
    shutil.copytree('../media','./media')
