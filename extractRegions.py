# extracting from region file
import os
import numpy as np
from numpy import array, newaxis, expand_dims
import csv
import pandas as pd
import subprocess


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
    arr_matrix= arr_matrix.transpose(0, 2, 1)
    print(arr_matrix)
    return(arr_matrix)


def write_temp_region(chr,s1,e1):
    with open('temp_region.txt', 'w') as tmp_file:
        tmp_file.write(str(chr) + "\t" + str(s1) + "\t" + str(e1))


def execute_shell():
    with open('temp_bash.sh', 'w') as the_file:
        the_file.write(
            "bedtools getfasta -fi /mnt/jLab/TT/Zea_mays_B73_v4.fasta -bed temp_region.txt > extracted_region.txt" )
    subprocess.call(["sh", "./temp_bash.sh"])


def read_chr_csv(file):

    chunksize = 2
    regions = []
    for chunk in pd.read_csv(file, sep="\t", header=None, skiprows=1,
                             chunksize=chunksize):

        regions = np.array(chunk)

        for item in regions:
            #print(chunk[item][0:5])
            chr, s1, e1, s2, e2 = item[0:5]
            write_temp_region(chr, s1, e1)
            execute_shell()
            train1 = matrix_fasta_regions()
            print(train1.shape)

        #print(regions[0][3])
        #print(regions[0][4])






read_chr_csv("/mnt/jLab/SERVER/geneMatrix/test.csv")