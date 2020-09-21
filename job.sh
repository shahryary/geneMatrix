#!/bin/sh
for entry in "csvFiles"/*.csv
do
	  python make_hfile.py  "$entry"
done
