"""Configuration file for the Sphinx documentation builder."""

project = 'Pyrfume'
copyright = '2019-, Rick Gerkin and Joel Mainland'
author = 'Rick Gerkin'
version = '0.1'
release = '0.1 alpha'
extensions = ['sphinxcontrib.apidoc',
              'sphinx.ext.intersphinx',
              'sphinx.ext.viewcode',
              'sphinx.ext.githubpages']
apidoc_module_dir = '../../pyrfume'
apidoc_output_dir = '../build'
apidoc_excluded_paths = ['*unit_test*']
apidoc_separate_modules = True
master_doc = 'index'
exclude_patterns = ['*unit_test*']
html_theme = 'alabaster'
