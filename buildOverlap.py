# building overlap between s2,e1
import pandas as pd
import time, os, glob
import traceback
import subprocess



def write_file(regios):
    regios.to_csv("tmp_output.bed", sep='\t', index=False, header=False, encoding='utf-8')
    with open('temp_bash.sh', 'w') as the_file:
        the_file.write("sort-bed tmp_output.bed | bedops --merge - > tmp_overlap.bed")
    subprocess.call(["sh", "./temp_bash.sh"])

# make regions between windows size, export as csv file

def find_overlap(intFile, seFile):
    extracted_region = []
    # reading bed file
    df_se = pd.read_csv(seFile, sep="\t", header=None, names=["chr", "s1", "e1"]).sort_values("s1")
    df_inter = pd.read_csv(intFile, sep="\t", header=None, names=["chr", "s1", "e1", "s2", "e2"],
                           usecols=[0,1,2,3,4]).sort_values("s2")
    # find records between the range.
    start_time = time.time()
    try:
        for index, row in df_se.iterrows():
            tmp_records = df_inter[(df_inter['s1'] >= row[1]) & (df_inter['e1'] <= row[2])]
            if not tmp_records.empty:
                # find overlap from tmp_records
                write_file(tmp_records[['chr', 's2', 'e2']])
                # reading tmp_overlap
                overlap = pd.read_csv("tmp_overlap.bed", sep="\t", header=None, names=["chr", "s2", "e2"]).sort_values("s2")
                # add original regions into data-frame
                tmp_records = overlap.assign(s1=row['s1'], e1=row['e1'])
                # append into main data-frame
                extracted_region.append(tmp_records)
            else:
                pass
            print(f'Current record: {index+1}, out of: {len(df_se)}')
        # concatenate data-frames
        extracted_region = pd.concat(extracted_region)
        # reset index
        extracted_region = extracted_region.reset_index(drop=True)
        # reordering columns
        cols = extracted_region.columns.tolist()
        cols = cols[0:1] + cols[-2:] + cols[1:3]
        extracted_region = extracted_region[cols]
        # write into file (csv format)
        extracted_region.to_csv("output_overlaps.csv", sep='\t', index=False, encoding='utf-8')
        print("Total records: ", len(extracted_region))
        print("--- %s seconds ---" % (time.time() - start_time))
    except Exception as e:
        print("type error: " + str(e))
        print(traceback.format_exc())


if __name__ == '__main__':
    directory = ''
    mainInteract="/home/yadi/jlab/Githubs/test_data/interacted.csv"
    s1e1File="/home/yadi/jlab/Githubs/test_data/out.csv"
    find_overlap(mainInteract, s1e1File)

