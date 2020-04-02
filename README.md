# Pyrfume

![Pyrfume logo](https://avatars3.githubusercontent.com/u/34174393?s=200&v=4)

#### `pyrfume` is a python library for olfactory psychophysics research. See "notebooks" for examples of use.

### Examples:
#### Note: these require the Pyrfume data library, provided separately.
```
# Get raw data from the Sigma Fragrance & Flavor Catalog
from pyrfume import sigma_ff
descriptors, data = sigma_ff.get_data()

# Get a PubChem CID-indexed dataframe of the odorant and descriptor data from that catalog:
import pyrfume
sigma = pyrfume.load_data('sigma/sigma.csv')
```
### [Website](http://pyrfume.org)

### [Docs](https://pyrfume.readthedocs.io/)
