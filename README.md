# Pyrfume

![Pyrfume logo](https://avatars3.githubusercontent.com/u/34174393?s=200&v=4)

#### `pyrfume` is a python library for olfactory psychophysics research. See "notebooks" for examples of use.
[![Python package](https://github.com/pyrfume/pyrfume/actions/workflows/pythonpackage.yml/badge.svg)](https://github.com/pyrfume/pyrfume/actions/workflows/pythonpackage.yml)
[![Travis](https://travis-ci.org/pyrfume/pyrfume.svg?branch=master)](https://travis-ci.org/pyrfume/pyrfume) 
[![Coverage Status](https://coveralls.io/repos/github/pyrfume/pyrfume/badge.svg?branch=master)](https://coveralls.io/github/pyrfume/pyrfume?branch=master)
![Zenodo](https://user-images.githubusercontent.com/549787/165869234-79bf95db-0b6c-495c-a1a8-b3db751f3352.png)


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
stimuli = pyrfume.load_data('bushdid_2014/stimuli.csv')
```

### Contributing

Just run `./develop.sh` to get started with developing `pyrfume`.

### [Website](http://pyrfume.org)

### [Data Repository](https://github.com/pyrfume/pyrfume-data)

### [Paper](https://www.biorxiv.org/content/10.1101/2022.09.08.507170)

### [Data Curation Status](http://status.pyrfume.org)

### [Docs](http://docs.pyrfume.org)

*Licensing/Copyright*: Data is provided as-is.  Licensing information for individual datasets is available in the data repository.  Takedown requests for datasets may be directed to admin at pyrfume dot org.  
