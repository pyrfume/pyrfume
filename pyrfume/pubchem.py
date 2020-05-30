import json

import requests


def get_summary(cid):
    url = "https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/cid/%d/description/json" % cid
    result = requests.get(url)
    summary = json.loads(result.content)
    return summary


def parse_summary_for_odor(summary):
    statements = []
    # keywords should include aroma but exclude aromatic (due to its special meaning in chemistry)
    keywords = ("odor", "odour", "smell", "aroma ", "aroma,", "aroma.", "fragrance")
    if "InformationList" in summary:
        for item in summary["InformationList"]["Information"]:
            if "Description" in item:
                for statement in item["Description"].split("."):
                    if any([x in statement.lower() for x in keywords]):
                        statements.append(statement.strip())
    return statements


def get_physical_description(cid):
    url = (
        "https://pubchem.ncbi.nlm.nih.gov/rest/pug_view/data/compound/%d/JSON?heading="
        "Physical+Description"
        % cid
    )
    result = requests.get(url)
    try:
        summary = json.loads(result.content)
    except UnicodeDecodeError:
        summary = {}
    return summary


def parse_physical_description_for_odor(physical_description):
    statements = []
    try:
        strings = [
            x["Value"]["StringWithMarkup"][0]["String"]
            for x in physical_description["Record"]["Section"][0]["Section"][0]["Section"][0][
                "Information"
            ]
        ]
    except KeyError:
        pass
    else:
        # keywords should include aroma but exclude aromatic
        # (due to its special meaning in chemistry)
        keywords = ("odor", "odour", "smell", "aroma ", "aroma,", "aroma.", "fragrance")
        for string in strings:
            for statement in string.split("."):
                if any([x in statement.lower() for x in keywords]):
                    statements.append(statement.strip())
    return statements


def get_ghs_classification(cid):
    url = (
        "https://pubchem.ncbi.nlm.nih.gov/rest/pug_view/data/compound/%d/JSON?heading=GHS"
        "+Classification"
        % cid
    )
    result = requests.get(url)
    try:
        summary = json.loads(result.content)
    except UnicodeDecodeError:
        summary = {}
    return summary


GHS_CODES = {
    "H300": "Fatal if swallowed.",
    "H301": "Toxic if swallowed",
    "H302": "Harmful if swallowed",
    "H303": "May be harmful if swallowed",
    "H304": "May be fatal if swallowed and enters airways",
    "H305": "May be harmful if swallowed and enters airways",
    "H310": "Fatal in contact with skin",
    "H311": "Toxic in contact with skin",
    "H312": "Harmful in contact with skin",
    "H313": "May be harmful in contact with skin",
    "H314": "Causes severe skin burns and eye damage",
    "H315": "Causes skin irritation",
    "H316": "Causes mild skin irritation",
    "H317": "May cause an allergic skin reaction",
    "H318": "Causes serious eye damage",
    "H319": "Causes serious eye irritation",
    "H320": "Causes eye irritation",
    "H330": "Fatal if inhaled",
    "H331": "Toxic if inhaled",
    "H332": "Harmful if inhaled",
    "H333": "May be harmful if inhaled",
    "H334": "May cause allergy or asthma symptoms or breathing difficulties if inhaled",
    "H335": "May cause respiratory irritation",
    "H336": "May cause drowsiness or dizziness",
    "H340": "May cause genetic defects",
    "H341": "Suspected of causing genetic defects",
    "H350": "May cause cancer",
    "H351": "Suspected of causing cancer",
    "H360": "May damage fertility or the unborn child",
    "H361": "Suspected of damaging fertility or the unborn child",
    "H361d": "Suspected of damaging the unborn child",
    "H361e": "May damage the unborn child",
    "H361f": "Suspected of damaging fertility",
    "H361g": "may damage fertility",
    "H362": "May cause harm to breast-fed children",
    "H370": "Causes damage to organs",
    "H371": "May cause damage to organs",
    "H372": "Causes damage to organs through prolonged or repeated exposure",
    "H373": "May cause damage to organs through prolonged or repeated exposure",
}


def parse_ghs_classification_for_odor(
    ghs_info,
    codes=("H330", "H331", "H334", "H340", "H350", "H350i", "H351", "H36", "H37"),
    only_percent=True,
    code_only=True,
):
    strings = []
    if "Record" in ghs_info:
        for block in ghs_info["Record"]["Section"][0]["Section"][0]["Section"][0]["Information"]:
            if block["Name"] == "GHS Hazard Statements":
                for entry in block["Value"]["StringWithMarkup"]:
                    string = entry["String"]
                    for code in codes:
                        match = (code + " (") if only_percent else code
                        if match in string:
                            if code_only:
                                string = string.split(":")[0]
                            strings.append(string)
    return strings
