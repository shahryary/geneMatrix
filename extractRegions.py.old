# extracting from region file
import os
import numpy as np
import time
import pandas as pd
import subprocess
import traceback
import h5py
import os
import numpy as np
from keras.models import load_model
np.set_printoptions(suppress=True)


def write_temp_region(chrm, s1, e1):
    with open('temp_region.txt', 'w') as tmp_file:
        tmp_file.write(str(chrm) + "\t" + str(s1) + "\t" + str(e1))


def execute_shell():
    with open('temp_bash.sh', 'w') as the_file:
        the_file.write(
            "bedtools getfasta -fi /home/yadi/Zea_mays_B73_v4.fasta -bed temp_region.txt > extracted_region.txt")
    subprocess.call(["sh", "./temp_bash.sh"])


def matrix_fasta_regions():
    arr_matrix = np.zeros((4, 5000))  # ACGT
    fp = open("extracted_region.txt")
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
    return arr_matrix


def prediction(test1, test2):

    pred_labels = model.predict([test1, test2])
    for item in zip(pred_labels):
        if np.round(item) == 1:
            print(item, "Item interacted")
            result = 1
        else:
            print(item)
            result = 0

    return result


def run_prediction(file):
    chunk_size = 3
    tmp_records = []
    start_time = time.time()
    try:
        for chunk in pd.read_csv(file, sep="\t", header=None, skiprows=1,
                                 chunksize=chunk_size):

            regions = np.array(chunk)
            print("running for chunk size: ", len(regions))
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
                result = prediction(test1, test2)
                # ------------------------------
                # write into file + prediction column
                # ------------------------------
                tmp_records.append([chrm, s1, e1, s2, e2, result])
        df = pd.DataFrame(tmp_records, columns=['chr', 's1', 'e1', 's2', 'e2', 'predict'])
        df = df.reset_index(drop=True)
        print("Writing into file ... ")
        df.to_csv("predicted.csv", sep='\t', index=False, encoding='utf-8')
        print("--- %s seconds ---" % (time.time() - start_time))
    except Exception as e:
        print("type error: " + str(e))
        print(traceback.format_exc())


if __name__ == '__extractRegions__':
    model = load_model('/data/logs/fulltrain/weights-improvement-15.hdf5')
    run_prediction("/home/yadi/jlab/Githubs/geneMatrix/test.csv")