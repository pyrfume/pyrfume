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

bushdid_data = bushdid.load_data()
bushdid_cas_list = bushdid_data.index.unique()
results = odorants.get_cids(bushdid_cas_list, kind='name', verbose=False)

bushdid_data = pd.Series(results, name='CID').to_frame().join(bushdid_data)
bushdid_data.head()

# Create a new file with CIDs and store here in `cids` dictionary
file_path = os.path.join(pyrfume.DATA, 'bushdid', 'bushdid.csv')
bushdid_data.to_csv(file_path)
