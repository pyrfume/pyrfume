# Pyrfume

![Pyrfume logo](https://avatars3.githubusercontent.com/u/34174393?s=200&v=4)

#### `pyrfume` is a python library for olfactory psychophysics research. See "notebooks" for examples of use.
[![Travis](https://travis-ci.org/pyrfume/pyrfume.svg?branch=master)](https://travis-ci.org/pyrfume/pyrfume) 
[![Coverage Status](https://coveralls.io/repos/github/pyrfume/pyrfume/badge.svg?branch=master)](https://coveralls.io/github/pyrfume/pyrfume?branch=master)

### Examples:
```
# Load data for Snitz et al, 2013 (PLOS Computational Biology)
import pyrfume
behavior = pyrfume.load_data('snitz_2013/behavior.csv')
molecules = pyrfume.load_data('snitz_2013/molecules.csv')

# Load data for Bushdid et al, 2014 (Science)
import pyrfume
behavior = pyrfume.load_data('bushdid_2014/behavior.csv')
molecules = pyrfume.load_data('bushdid_2014/molecules.csv')
mixtures = pyrfume.load_data('bushdid_2014/behavior.csv')
```

### [Website](http://pyrfume.org)

### [Data Curation Status](http://status.pyrfume.org)

### [Docs](https://pyrfume.readthedocs.io/)
