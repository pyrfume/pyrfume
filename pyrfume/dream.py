"""Utilities for DREAM challenge data"""

import numpy as np

import pyrfume


def get_descriptors(format=False):
    """Get DREAM challenge descriptors"""
    if format:
        result = [
            "Intensity",
            "Pleasantness",
            "Bakery",
            "Sweet",
            "Fruit",
            "Fish",
            "Garlic",
            "Spices",
            "Cold",
            "Sour",
            "Burnt",
            "Acid",
            "Warm",
            "Musky",
            "Sweaty",
            "Ammonia",
            "Decayed",
            "Wood",
            "Grass",
            "Flower",
            "Chemical",
        ]
    else:
        result = [
            "INTENSITY/STRENGTH",
            "VALENCE/PLEASANTNESS",
            "BAKERY",
            "SWEET",
            "FRUIT",
            "FISH",
            "GARLIC",
            "SPICES",
            "COLD",
            "SOUR",
            "BURNT",
            "ACID",
            "WARM",
            "MUSKY",
            "SWEATY",
            "AMMONIA/URINOUS",
            "DECAYED",
            "WOOD",
            "GRASS",
            "FLOWER",
            "CHEMICAL",
        ]
    return result


def get_cids():
    """Get PubChem IDs for DREAM challenge molecules"""
    df = pyrfume.load_data("keller_2017/cids.csv", index_col=None)
    return list(df["CID"])


def dilution_to_magnitude(dilution):
    denom = dilution.replace('"', "").replace("'", "").split("/")[1].replace(",", "")
    return np.log10(1.0 / float(denom))
