#!/usr/bin/env python

# Making h5 file from regions
import os
import numpy as np
import time
import pandas as pd
import subprocess
import traceback
import h5py
import sys
np.set_printoptions(suppress=True)

def write_temp_region(chrm, s1, e1):
    with open('temp_region-'+file_name+'.txt', 'w') as tmp_file:
        tmp_file.write(str(chrm) + "\t" + str(s1) + "\t" + str(e1))


def execute_shell():
    with open('temp_bash-'+file_name+'.sh', 'w') as the_file:
        the_file.write(
            "bedtools getfasta -fi /home/yadi/DP/geneMatrix/Zea_mays_B73_v4.fasta -bed temp_region-"+file_name+".txt > extracted_region-"+file_name+".txt")
    subprocess.call(["sh", "./temp_bash-"+file_name+".sh"])


def matrix_fasta_regions():
    arr_matrix = np.zeros((4, 5000))  # ACGT
    fp = open("extracted_region-"+file_name+".txt")
    for i, line in enumerate(fp):
        if i == 1:
            line_character_count = 0
            for c in line:
                # print c
                if c == "A":
                    arr_matrix[0][line_character_count] = 1
                if c == "C":
                    arr_matrix[1][line_character_count] = 1
                if c == "G":
                    arr_matrix[2][line_character_count] = 1
                if c == "T":
                    arr_matrix[3][line_character_count] = 1
                line_character_count += 1
    fp.close()

    arr_matrix = arr_matrix.astype('int')
    return arr_matrix


def run_prediction(file):
    chunk_size = 100000
    tmp_records = []
    start_time = time.time()
    # --------------------------------
    num_lines = sum(1 for line in open(file))
    # --------------------------------
    h5_filename = "/mnt/intStorage/regions_matrics.h5"
    if not os.path.isfile(h5_filename):
        hf = h5py.File(h5_filename, 'w')
        hf.create_dataset('region_s1_e1', shape=(num_lines-1, 4, 5000), dtype="int",
                          data=None)
        hf.create_dataset('region_s2_e2', shape=(num_lines-1, 4, 5000), dtype="int",
                          data=None)
    # --------------------------------
    index = 0
    try:
        for chunk in pd.read_csv(file, sep="\t", header=None, skiprows=1,
                                 chunksize=chunk_size):

            regions = np.array(chunk)
            print("Running for chunk size: ", len(regions))
            for item in regions:
                chrm, s1, e1, s2, e2 = item[0:5]
                # ---- making matrix for first set s1,e1
                write_temp_region(chrm, s1, e1)
                execute_shell()
                test1 = matrix_fasta_regions()
                # ---- making matrix for second set s2,e2
                write_temp_region(chrm, s2, e2)
                execute_shell()
                test2 = matrix_fasta_regions()
                # -----------------------------
                # Predicting from Model
                # -----------------------------
                with h5py.File(h5_filename, 'a') as hf:
                    hf["region_s1_e1"][index] = test1
                    hf["region_s2_e2"][index] = test2
                index += 1
                # ------------------------------
                print("Number of records: ", index, "out of: ", num_lines-1)
                # ------------------------------
                #tmp_records.append([chrm, s1, e1, s2, e2, prob, result])
        print("Writing into file ... ")
        print("--- %s seconds ---" % (time.time() - start_time))
    except Exception as e:
        print("type error: " + str(e))
        print(traceback.format_exc())



file_name = os.path.basename(os.path.splitext(sys.argv[1])[0])
os.chdir("csvFiles/")
run_prediction(file_name+".csv")

