#!/usr/bin/env bash
# shellcheck disable=SC1091

dir_root=$(dirname "$(readlink -f "$0")")

# Create virtual environment
virtualenv "$dir_root/.venv"
source "$dir_root/.venv/bin/activate"

# Install pyrfume
if [ "$1" == "pip" ]
then
  pip install pyrfume
else
  pip install -e .
fi

# Test import of installed pyrfume
cd /tmp || exit
python -c "import pyrfume"
python -c "
    import sys
    try:
        from pyrfume import optimization
        sys.exit(2)
    except ImportError:
        pass
"
# Cleanup environment
cd "$dir_root" || exit
rm -rf "$dir_root/.venv"

