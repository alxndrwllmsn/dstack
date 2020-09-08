#!/bin/bash

echo "Run all unittest for dstcak:\n"

echo "Unittest: cim module"
python3 cimtest.py
echo
echo "Unittest: cimutil module"
python3 cimutiltest.py
echo
echo "Unittest: msutil module"
python3 msutiltest.py
