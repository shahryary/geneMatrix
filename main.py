#
import pandas as pd
import sys


def matrix(minWinSize, maxWinSize):
    main_region = []
    original_file = []
    tmp_pos = []

    # reading bed file
    '''
    with open("/home/yadi/Downloads/1_final.bed")as f:
        for line in f:
            original_file.append(line.strip().split())
    # convert into integer
    #original_file = map(int, original_file)
    original_file = [list(map(int, lst)) for lst in original_file]
    df = pd.DataFrame(original_file)
    '''
    df = pd.read_csv('/home/yadi/Downloads/1_final.bed', sep="\t", header=None, names=["a", "b", "c"])

    for index, row in df.iterrows():
        c = df[(df['b'] >= row[2]+minWinSize) & (df['c'] <= row[2] + maxWinSize)]
        print(f'{c}')  # Press Ctrl+F8 to toggle the breakpoint.
        if c.empty:
            sys.exit()


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    matrix(20000, 140000)

