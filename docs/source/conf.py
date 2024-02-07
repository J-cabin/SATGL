# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information
from recommonmark.parser import CommonMarkParser
import sphinx_rtd_theme
import recommonmark
from recommonmark.transform import AutoStructify


project = 'SATGL'
copyright = '2024, BUPT-GAMMA LAB'
author = 'BUPT-GAMMA LAB'
release = '0.1'
github_doc_root = 'https://github.com/rtfd/recommonmark/tree/master/docs/'

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    'recommonmark',
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
    "sphinx_copybutton",
    'sphinx.ext.autodoc',
    'sphinx.ext.autosummary',
    'sphinx.ext.doctest',
    'sphinx.ext.githubpages',
]

autosummary_generate = True
templates_path = ['_templates']
exclude_patterns = []



# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

# html_theme = 'alabaster'
# html_static_path = ['_static']
html_theme = "sphinx_rtd_theme"
html_theme_path = [sphinx_rtd_theme.get_html_theme_path()]

html_theme_options = {
    'collapse_navigation': False,
    'display_version': True,
    'logo_only': True,
    'navigation_depth': 2,
}

source_parsers = {
 '.md': CommonMarkParser,
}
source_suffix = ['.rst', '.md']

def setup(app):
    app.add_config_value('recommonmark_config', {
        'url_resolver': lambda url: github_doc_root + url,
        'auto_toc_tree_section': 'Contents',
            }, True)
    app.add_transform(AutoStructify)
