# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = 'GDTchron'
copyright = '2025, Dylan Vasey, Peter Scully, John Naliboff, Sascha Brune'
author = 'Dylan Vasey, Peter Scully, John Naliboff, Sascha Brune'
release = '0.1.2'

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

nbsphinx_execute = 'never'

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
import glob
shutil.copy('../CONTRIBUTING.md', './CONTRIBUTING.md')
shutil.copy('../README.md', './README.md')

# Delete logo from README
with open('README.md','r') as file:
    lines = file.readlines()

with open('README.md','w') as file:
    file.writelines(lines[3:])

# Copy logo
try:
    shutil.copytree('../media','./media')
except FileExistsError:
    shutil.rmtree('./media')
    shutil.copytree('../media','./media')

# Copy notebooks
# Copy GDMATE notebooks into the docs directory
nbdir = '../notebooks'
docdir = './notebooks'

try:
    shutil.copytree(nbdir,docdir)
except FileExistsError:
    shutil.rmtree(docdir)
    shutil.copytree(nbdir,docdir)

# Copy ASPECT notebooks into the docs directoy
anb_files = '../aspect/*.ipynb'

for filepath in glob.glob(anb_files):
    shutil.copy(filepath, docdir)