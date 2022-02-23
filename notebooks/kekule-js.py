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

from IPython.display import display, Javascript, HTML

# Load the Kekule JS file and CSS style
# These can be downloaded from:
# http://partridgejiang.github.io/Kekule.js/download/files/kekule.js.zip
Javascript(filename='kekule/kekule.min.js',
           css='kekule/themes/default/kekule.css')

# Make sure the mol file we want to show exists
import os
assert os.path.isfile('../data/random.mol')

# Create a canvas on which to show the molecule
HTML('<div id="div1"></div>')

# Show the molecule
Javascript("""
// read the file
fetch('../data/random.mol')
  .then(function(response) {
    return response.text();
  })
  .then(function(text) {
    // Show the contents of the mol file in the js console
    console.log(text);
    // make a Kekule molecule object
    var mol = Kekule.IO.loadFormatData(text, 'mol');
    // Make a viewer and bind it to the <div> element above
    var chemViewer = new Kekule.ChemWidget.Viewer(document.getElementById('div1'));
    // Put the molecule in the viewer
    chemViewer.setChemObj(mol);
  });
""")
