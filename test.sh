pip install -q coveralls
UNIT_TEST_SUITE="pyrfume.unit_test buffer"
coverage run -m --source=. --omit=*unit_test*,setup.py,.eggs $UNIT_TEST_SUITE
