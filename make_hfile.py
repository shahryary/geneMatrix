#!/usr/bin/env python

# validating interacts in memory based on CPU/GPU
import os
import time
import pandas as pd
import traceback
import numpy as np
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
    arr_matrix = arr_matrix.astype('int')
    return arr_matrix


def prediction(test1, test2):
    pred_result = []
    pred_labels = model.predict([test1, test2])
    for item in zip(pred_labels):
        if np.round(item) == 1:
            result = 1
        else:
            result = 0
        pred_result.append([result, "%.8f" % item[0][0]])
    return pred_result


def ext_first(lst):
    return [item[1] for item in lst]


def ext_second(lst):
    return [item[0] for item in lst]


def run_prediction(csvfile, chrm):
    num_lines = sum(1 for line in open(csvfile))
    chunk_size = 100000
    first_record = True
    tmp_s1 = []
    tmp_e1 = []
    count = 0
    file_number = 0
    try:
        for chunk in pd.read_csv(csvfile, sep="\t", header=None, skiprows=1,
                                 chunksize=chunk_size):
            start_time = time.time()
            test1 = []
            test2 = []
            A = []
            B = []
            main_predict=[]
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
                tmp_records.append([chrm, tmp_s1, tmp_e1, item[3], item[4]])

            A = np.array(A).transpose(0, 2, 1)
            B = np.array(B).transpose(0, 2, 1)
            print("\n--- Computing predictions ... ---")
            result = prediction(A, B)
            main_predict.append(result)
            first_record = True
            print("--- Processed: ", count, "out of: ", num_lines)
            main_predict = list(chain.from_iterable(main_predict))
            dataframe = pd.DataFrame(tmp_records, columns=['chr', 's1', 'e1', 's2', 'e2'])
            dataframe['prob'] = ext_first(main_predict)
            dataframe['interacted'] = ext_second(main_predict)
            dataframe = dataframe.reset_index(drop=True)
            print("--- Writing into file ... ---- ")
            file_number += 1
            dataframe.to_csv(output + folder_name + "/part-" + str(file_number) + "-" + str(count) + "-" + folder_name +
                             ".csv", sep='\t', index=False, encoding='utf-8')
            print("--- %s seconds ---" % (time.time() - start_time))
            print(15*"-")

    except Exception as e:
        print("type error: " + str(e))
        print(traceback.format_exc())


hdf_file = "/mnt/intStorage/deeplearning/myDP/logs/fulltrain/"
chr_dir = "/home/yadi/DP/geneMatrix/chr1/"
output = "/mnt/intStorage/output/"

model = load_model(hdf_file+'weights-improvement-15.hdf5')

csv_file = sys.argv[1]
folder_name = os.path.basename(os.path.splitext(csv_file)[0])

create_dir = output + folder_name
if not os.path.exists(create_dir):
    try:
        os.mkdir(create_dir)
    except OSError:
        print("Creation of the directory failed")

run_prediction(csv_file, chrm="1")

