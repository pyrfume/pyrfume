import requests

url = "https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/cid/8857/property/MolecularWeight,IsomericSMILES/JSON"
response = request.get(url)
print(response.status_code)
print(response.content)
