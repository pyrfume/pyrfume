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

file_path = os.path.join(pyrfume.DATA, 'snitz', 'Snitz144.csv')
snitz_data_raw = pd.read_csv(file_path)

results = odorants.get_cids(snitz_data_raw['CAS'], kind='name', verbose=False)

snitz_data = pd.Series(results, name='CID').to_frame().join(snitz_data_raw.set_index('CAS'))
snitz_data.head()

file_path = os.path.join(pyrfume.DATA, 'snitz', 'snitz.csv')
snitz_data.to_csv(file_path)
