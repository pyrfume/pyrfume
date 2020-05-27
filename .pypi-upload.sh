VERSION=`python -c "import pyrfume; print(pyrfume.__version__)"`
python setup.py sdist build
twine upload dist/pyrfume-${VERSION}.tar.gz

