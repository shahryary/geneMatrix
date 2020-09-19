#!/usr/bin/env python

# extracting from region file
import os
import time
import pandas as pd
import subprocess
import traceback
import numpy as np
import csv
import sys
from keras.models import load_model
from itertools import chain

np.set_printoptions(suppress=True)


def matrix_fasta_regions(tmp_one, tmp_two):
    arr_matrix = np.zeros((4, 5000))  # ACGT
    fp = open(chr_dir + str(tmp_one) + "_" + str(tmp_two) + ".txt")
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

    #arr_matrix = arr_matrix.reshape((1,) + arr_matrix.shape)
    arr_matrix = arr_matrix.astype('int')
    #arr_matrix = arr_matrix.transpose(0, 2, 1)
    return arr_matrix


def prediction(test1, test2):
    pred_result = []
    pred_labels = model.predict([test1, test2])
    for item in zip(pred_labels):
        if np.round(item) == 1:
            print(item, "Item interacted")
            result = 1
        else:
            print(item)
            result = 0
        pred_result.append([result, "%.8f" % item[0][0]])
    return (pred_result)


def ext_first(lst):
    return [item[1] for item in lst]

def ext_second(lst):
    return [item[0] for item in lst]


def run_prediction(file, chrm):
    num_lines = sum(1 for line in open(file))
    chunk_size = 2

    start_time = time.time()
    first_record = True
    tmp_s1 = []
    tmp_e1 = []

    count = 0
    try:
        for chunk in pd.read_csv(file, sep="\t", header=None, skiprows=1,
                                 chunksize=chunk_size):

            test1 = []
            test2 = []
            A = []
            B = []
            main_predict = []
            tmp_records = []
            regions = np.array(chunk)
            for item in regions:
                if first_record:
                    tmp_s1 = item[1]
                    tmp_e1 = item[2]
                    test1 = matrix_fasta_regions(tmp_s1, tmp_e1)
                    first_record = False

                if tmp_s1 != item[1] and tmp_e1 != item[2]:
                    test1 = matrix_fasta_regions(item[1], item[2])
                    tmp_s1 = item[1]
                    tmp_e1 = item[2]

                test2 = matrix_fasta_regions(item[3], item[4])
                A.append(test1)
                B.append(test2)
                count += 1
                # ------------------------------
                print("item: ", count, "out of: ", num_lines)
                # ------------------------------
                tmp_records.append([chrm, tmp_s1, tmp_e1, item[3], item[4]])

            A = np.array(A).transpose(0, 2, 1)
            B = np.array(B).transpose(0, 2, 1)
            result = prediction(A, B)
            main_predict.append(result)
            first_record = True
            print("--- Proccessed: ", count, "out of: ", num_lines)
            main_predict = list(chain.from_iterable(main_predict))
            dataframe = pd.DataFrame(tmp_records, columns=['chr', 's1', 'e1', 's2', 'e2'])
            dataframe['prob'] = ext_first(main_predict)
            dataframe['interacted'] = ext_second(main_predict)
            dataframe = dataframe.reset_index(drop=True)
            print("Writing into file ... ")
            dataframe.to_csv(output+"predicted-" + str(count) + ".csv", sep='\t', index=False, encoding='utf-8')
            print("--- %s seconds ---" % (time.time() - start_time))

    except Exception as e:
        print("type error: " + str(e))
        print(traceback.format_exc())


hdf_file = "/mnt/jLab/SERVER/testing/"
chr_dir = "/mnt/jLab/SERVER/testing/chr1/"
output = "/mnt/jLab/SERVER/testing/"

#os.chdir("csvFiles/")

model = load_model(hdf_file+'weights-improvement-15.hdf5')
#file_name = os.path.basename(os.path.splitext(sys.argv[1])[0])
file_name = "/mnt/jLab/SERVER/testing/test"
run_prediction(file_name+".csv", chrm="1")
