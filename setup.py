"""Setup file for Pyrfume"""

from pathlib import Path
from setuptools import setup, find_packages


def read_requirements():
    """Parses requirements from requirements.txt"""
    reqs_path = Path(__file__).parent / 'requirements.txt'
    reqs = None
    with open(reqs_path) as reqs_file:
        reqs = reqs_file.read().splitlines()
    return reqs


def get_version():
    version = {}
    with open(Path(__file__).parent / 'pyrfume' / 'version.py') as f:
        exec(f.read(), version)
    return version['__version__']


readme_path = Path(__file__).parent / 'README.md'
with open(readme_path, encoding='utf-8') as f:
    long_description = f.read()

    
setup(
    name='pyrfume',
    version=get_version(),
    author='Rick Gerkin',
    author_email='rgerkin@asu.edu',
    packages=find_packages(),
    url='http://pyrfume.scidash.org',
    license='MIT',
    long_description_content_type='text/markdown',
    long_description=long_description,
    description='A validation library for human olfactory psychophysics research.',
    install_requires=read_requirements(),
    extras_require={'features': ['rdkit']}
)
