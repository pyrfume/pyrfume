#!/bin/bash
eval "$(conda shell.bash hook)"
conda create -y --name pyrfume python
conda activate pyrfume
if [ "$1" == "pip" ]
then
  pip install pyrfume
else
  pip install -e .
fi
cd $HOME
python -c "import pyrfume"
conda deactivate
conda env remove --name pyrfume
