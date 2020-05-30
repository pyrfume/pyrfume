import pathlib

import pyrfume

MAINLAND_INTENSITY_PATH = pathlib.Path("mainland_intensity")


def load_mainland_data(extra_cols=[]):
    cols = ["Subject", "Odor", "CAS", "Concentration", "IntensityRating"]
    cols += extra_cols
    df = pyrfume.load_data(MAINLAND_INTENSITY_PATH / "all data Supra clean.csv", index_col=None)
    df = df[cols]
    df["Subject"] = df["Subject"].astype(int)
    df["IntensityRating"] /= 100
    return df


def load_threshold_compilation():
    """Load data from http://www.thresholdcompilation.com/paginas/odour/odour.html
    A license or purchased copy of the book is required to access this data"""
    pyrfume.load_data("threshold/parsed_threshold_data_in_air.csv")
