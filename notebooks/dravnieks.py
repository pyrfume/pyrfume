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

cas, descriptors, data = dravnieks.get_data()
drav = pd.DataFrame.from_dict(data).T.round(2)

# Turn CAS into CIDs
cas_list = list(data.keys())
results = odorants.get_cids(cas_list, kind='name', verbose=False)

drav = pd.Series(results, name='CID').to_frame().join(drav)
drav.head()

# Create a new file with CIDs and store here in `cids` dictionary
file_path = os.path.join(pyrfume.DATA, 'dravnieks', 'dravnieks.csv')
drav.to_csv(file_path)
