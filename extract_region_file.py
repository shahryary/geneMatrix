#!/usr/bin/env python

# extracting from region file
import os
import time
import pandas as pd
import subprocess
import traceback
import sys
import numpy as np

np.set_printoptions(suppress=True)

# extract into text file
# just sequnce into file


def write_temp_region(chrm, s1, e1):
    with open('temp_region.txt', 'w') as tmp_file:
        tmp_file.write(str(chrm) + "\t" + str(s1) + "\t" + str(e1))
        tmp_file.close()
    # create shell file and extract
    with open('temp_bash.sh', 'w') as the_file:
        the_file.write(
            "bedtools getfasta -fi Zea_mays_B73_v4.fasta -bed temp_region.txt > " + str(s1) + "_" + str(e1) + ".txt")
    subprocess.call(["sh", "./temp_bash.sh"])


def run_prediction(file):
    num_lines = sum(1 for line in open(file))
    count = 0
    start_time = time.time()
    try:
        df = pd.read_csv(file, sep="\t", header=None, skiprows=0)
        regions = np.array(df)
        for item in regions:
            print("running for chunk size: ", item)

            chrm, s1, e1 = item[0:3]
            # ---- making matrix for first set s1,e1
            write_temp_region(chrm, s1, e1)
            count += 1
            # ------------------------------
            print("item: ", count, "out of: ", num_lines)
            # ------------------------------
        print("Writing into file ... ")
        print("--- %s seconds ---" % (time.time() - start_time))
        os.remove("temp_region.txt")
        os.remove("temp_bash.sh")
    except Exception as e:
        print("type error: " + str(e))
        print(traceback.format_exc())


dir = "chr1"
if not os.path.exists(dir):
    try:
        os.mkdir(dir)
    except OSError:
        print("Creation of the directory failed")

os.chdir(dir)
run_prediction("/home/yadi/1.bed")
