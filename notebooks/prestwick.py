# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.10.3
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

file_path = os.path.join(pyrfume.DATA, 'PrestwickChemLib.smi')
prestwick_data = pd.read_csv(file_path, header=None, sep='\t')[0]
prestwick_data.head()

results = odorants.get_cids(prestwick_data['SMILES'], kind='SMILES', verbose=False)

prestwick_data = pd.Series(results, name='CID').to_frame().join(prestwick_data)[['CID']]
prestwick_data.head()

# Create a new file with CIDs and store here in `cids` dictionary
file_path = os.path.join(pyrfume.DATA, 'prestwick.csv')
prestwick_data.to_csv(file_path)
