#!/bin/bash

echo "Run all unittest for dstcak:\n"

echo "Unittest: cgrid module"
python3 cgrid_test.py
echo
echo "Unittest: cim module"
python3 cim_test.py
echo
echo "Unittest: cimutil module"
python3 cimutil_test.py
echo
echo "Unittest: msutil module"
python3 msutil_test.py
