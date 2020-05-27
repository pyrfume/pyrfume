import pyrfume

from . import dream


def load_raw_bmc_data(nrows=None):
    """Load raw data from Keller and Vosshall, 2016 supplement."""
    df_raw = pyrfume.load_data("keller_2016/12868_2016_287_MOESM1_ESM.xlsx", header=2)
    return df_raw


def format_bmc_data(
    df,  # The raw data frame returned by `load_raw_bmc_data`
    only_dream_subjects=False,  # Whether to only keep DREAM subjects
    only_dream_descriptors=False,  # Whether to only keep DREAM descriptors
    only_dream_molecules=False,
):  # Whether to only keep DREAM molecules
    """Format raw data from the BMC paper to be usable for modeling"""
    # Remove leading and trailing white space from column names
    df.columns = df.columns.str.strip()

    # Get the raw DREAM descriptor list
    descriptors_raw = dream.get_descriptors()
    # Get the publication-style descriptor names
    descriptors = dream.get_descriptors(format=True)
    # Revise to the Keller and Vosshall descriptor names
    descriptors_raw[0] = "HOW STRONG IS THE SMELL?"
    descriptors_raw[1] = "HOW PLEASANT IS THE SMELL?"

    # Possibly include "Familiarity" as a descriptor
    if not only_dream_descriptors:
        descriptors_raw.append("HOW FAMILIAR IS THE SMELL?")
        descriptors.append("Familiarity")

    # Possibly restrict subjects to those used in the DREAM challenge
    # Note that numeric subject IDs in the BMC paper and in the DREAM
    # challenge are not identical
    if only_dream_subjects:
        df["Subject"] = df["Subject # (DREAM challenge)"].fillna(0).astype(int)
        df = df[df["Subject"] > 0]
    else:
        df["Subject"] = df["Subject # (this study)"].astype(int)

    # Rename columns to match DREAM challenge
    df = df.rename(columns={"Odor dilution": "Dilution"})
    df = df.rename(columns=dict(zip(descriptors_raw, descriptors)))

    # Fix CIDs for molecules that only have CAS registry numbers.
    # Geranylacetone didn't have a CID listed in the raw data
    # Isobutyl acetate had the wrong CAS number in the raw data
    df["CID"] = (
        df["CID"]
        .astype(str)
        .str.replace("3796-70-1", "1549778")
        .str.replace("109-19-0", "8038")
        .astype(int)
    )

    # Possibly keep only the 476 DREAM challenge molecules
    if only_dream_molecules:
        dream_CIDs = dream.get_cids()
        assert len(dream_CIDs) == 476
        df = df[df["CID"].isin(dream_CIDs)]

    # Keep only relevant columns
    df = df[["CID", "Dilution", "Subject"] + descriptors]

    # Fill NaN descriptors values with 0 if Intensity is not 0.
    df = df.apply(lambda x: x.fillna(0) if x["Intensity"] > 0 else x, axis=1)

    # Make dilution values integer -log10 dilutions
    df["Dilution"] = df["Dilution"].apply(dream.dilution_to_magnitude).astype(float)

    # Set index and set column axis name
    df = df.set_index(["CID", "Dilution", "Subject"])
    df.columns.name = "Descriptor"

    # Identify replicates and add this information to the index
    df["Replicate"] = df.index.duplicated().astype(int)
    df = df.reset_index().set_index(["CID", "Dilution", "Replicate", "Subject"])
    if only_dream_subjects:
        # DREAM subjects replicates should be properly indexed now
        assert df.index.duplicated().sum() == 0

    # Rearrange dataframe to pivot subjects and descriptors
    df = df.unstack("Subject").stack("Descriptor")
    df = df.reorder_levels(["Descriptor", "CID", "Dilution", "Replicate"])
    df = df.sort_index()

    return df
