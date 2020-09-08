#
import pandas as pd
import time, os, glob
import traceback



def find_scope(bed_file, min_win_size, max_win_size, output):
    extracted_region = []
    # reading bed file
    df = pd.read_csv(bed_file, sep="\t", header=None, names=["chr", "s2", "e2"])
    # find records between the range.
    start_time = time.time()
    try:
        for index, row in df.iterrows():
            tmp_records = df[(df['s2'] >= row[2] + min_win_size) & (df['e2'] <= row[2] + max_win_size)]
            if not tmp_records.empty:
                # add original regions into data-frame
                tmp_records = tmp_records.assign(s1=row['s2'], e1=row['e2'])
                # append into main data-frame
                extracted_region.append(tmp_records)
                print(f'Current record: {index+1}, out of: {len(df)}')
            else:
                pass
        # concatenate data-frames
        extracted_region = pd.concat(extracted_region)
        # reset index
        extracted_region = extracted_region.reset_index(drop=True)
        # reordering columns
        cols = extracted_region.columns.tolist()
        cols = cols[0:1] + cols[-2:] + cols[1:3]
        extracted_region = extracted_region[cols]
        # write into file (csv format)
        extracted_region.to_csv(output+".csv", sep='\t', index=False, encoding='utf-8')
        print("Total records: ", len(extracted_region))
        print("--- %s seconds ---" % (time.time() - start_time))
    except Exception as e:
        print("type error: " + str(e))
        print(traceback.format_exc())


if __name__ == '__main__':
    directory = ''
    for filename in glob.glob(directory+"*.bed"):
        fname = os.path.splitext(os.path.basename(filename))[0]
        bedfile = filename
        find_scope(bedfile, 20000, 140000, fname)

