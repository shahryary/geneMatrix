#!/bin/sh
# spilit csv file into n  lines  
cat chr1.csv | parallel --header : --pipe -N6435280 'cat > file_{#}.csv'
