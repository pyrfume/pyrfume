"""Setup file for Pyrfume
Provided as a shim for tools which require a setup.py file,
including editable installs (e.g. `python _setup.py develop --user`)
"""

from pathlib import Path
from setuptools import setup, find_packages


setup(
    name='pyrfume',
    author='Rick Gerkin',
    author_email='rgerkin@asu.edu',
    packages=find_packages(),
    url='http://pyrfume.org',
    license='MIT',
    description='A validation library for human olfactory psychophysics research.',
)
