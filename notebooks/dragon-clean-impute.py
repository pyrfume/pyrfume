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

# ## Turn raw Dragon features into cleaned, imputed, features which are either min-max scaled or standardized

# +
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

from pyrfume import features
# -

df = features.load_dragon()
df.shape

df_cleaned = features.clean_features(df)
df_cleaned.shape

df_cleaned_minmaxed = features.scale_features(df_cleaned, 'minmax')
df_cleaned_minmaxed.shape

df_cleaned_standardized = features.scale_features(df_cleaned, 'standardize')
df_cleaned_standardized.shape

# %time df_cleaned_minmaxed_imputed = features.impute_features(df_cleaned_minmaxed)
df_cleaned_minmaxed_imputed.shape

features.save_dragon(df_cleaned_minmaxed_imputed, '-cleaned-minmaxed-imputed')

# %time df_cleaned_standardized_imputed = features.impute_features(df_cleaned_standardized)
df_cleaned_standardized_imputed.shape

features.save_dragon(df_cleaned_standardized_imputed, '-cleaned-standardized-imputed')


