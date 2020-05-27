import numpy as np

import pyrfume


def get_mainland(raw=False, vendors=None):
    """Return a dataframe containing odorants in Joel Mainland's cabinet"""
    df = pyrfume.load_data("cabinets/mainland.csv").set_index("CID")
    if not raw:
        if vendors:  # e.g. ['sigma']
            # Keep only odorants obtained from certain vendors
            df = df[df["SourcedFrom"].str.lower().isin(vendors)]

        # For odorants with no relative cost information
        # (probably discontinued), fill with a large number so they
        # don't get used.  Also use log10 $/mol
        df.loc[:, "$/mol"] = np.log10(df.loc[:, "$/mol"].fillna(1e15))

        # Fix price information
        df["Price"] = df["Price"].astype("str").apply(lambda x: x.replace("$", "")).astype("float")

        # Remove odorants with no price
        df = df.dropna(subset=["Price"])

        # Sort by price and then take only the cheapest instance of each CID
        df = df.sort_values("Price").groupby("CID").first()

        # Drop values with no CID
        df = df.loc[df.index > 0]

        # Sort CIDs from low to high
        df = df.sort_index()

    return df
