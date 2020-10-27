#!/usr/bin/env python

# validate given position that predicted correctly or not.
import os
import numpy as np
import time
import pandas as pd
import subprocess
import traceback
import h5py
import numpy as np
from keras.models import load_model
np.set_printoptions(suppress=True)

import sys


def write_temp_region(chrm, s1, e1):
    with open('temp_region.txt', 'w') as tmp_file:
        tmp_file.write(str(chrm) + "\t" + str(s1) + "\t" + str(e1))


def execute_shell():
    with open('temp_bash.sh', 'w') as the_file:
        the_file.write(
            "bedtools getfasta -fi "+run_path+"Zea_mays_B73_v4.fasta -bed temp_region.txt >"+run_path+"extracted_region.txt")
    subprocess.call(["sh", "./temp_bash.sh"])


def matrix_fasta_regions():
    arr_matrix = np.zeros((4, 5000))  # ACGT
    fp = open(run_path+"extracted_region.txt")
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

    arr_matrix = arr_matrix.reshape((1,) + arr_matrix.shape)
    arr_matrix = arr_matrix.astype('int')
    arr_matrix = arr_matrix.transpose(0, 2, 1)
    print(arr_matrix)
    return arr_matrix


def prediction(test1, test2):

    pred_labels = model.predict([test1, test2])
    for item in zip(pred_labels):
        if np.round(item) == 1:
            #print(item, "Item interacted")
            result = 1
        else:
            #print(item)
            result = 0

    return (result, "%.8f"% item[0][0])


def run_prediction(csvfile):
    chunk_size = 100000
    tmp_records = []
    start_time = time.time()
    try:
        for chunk in pd.read_csv(csvfile, sep="\t", header=None, skiprows=1,
                                 chunksize=chunk_size):

            regions = np.array(chunk)
            print("running for chunk size: ", len(regions))
            for item in regions:
                print(item)
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
                result, prob = prediction(test1, test2)
                # ------------------------------
                # write into file + prediction column
                # ------------------------------
                tmp_records.append([chrm, s1, e1, s2, e2, prob, result])
        df = pd.DataFrame(tmp_records, columns=['chr', 's1', 'e1', 's2', 'e2', 'prob', 'interacted'])
        df = df.reset_index(drop=True)
        print(df)
        #df.to_csv("../output/predicted-"+file_name+".csv", sep='\t', index=False, encoding='utf-8')
        print("--- %s seconds ---" % (time.time() - start_time))
    except Exception as e:
        print("type error: " + str(e))
        print(traceback.format_exc())

run_path="/mnt/jLab/SERVER/testing/"

os.chdir(run_path)
model = load_model('/mnt/jLab/SERVER/testing/weights-improvement-15.hdf5')
#csv_file = sys.argv[1]
csv_file="/mnt/jLab/SERVER/testing/test.csv"
run_prediction(csv_file)

