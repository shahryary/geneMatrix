#!/bin/sh

# Merging all csvfiles inside of folders into one csv file. 
# copy to the root of folders 
#--------------- 
if [ ! -d merged_csv ]; then
	mkdir merged_csv
fi
#---------------

for folder in *f*/
do

		new_file=$(basename "$folder")
		echo "================================================="
		echo "merging files in folder: " $new_file

		head -1 $folder/*-100000-*.csv > $folder/$new_file.csv
		for file in $(ls -rt $folder/p*.csv)
		do
			echo "reading " $file
			tail -n +2 -q $file >> $folder/$new_file.csv		
		done
		mv $folder/$new_file.csv merged_csv/

		echo "number of lines: " $(wc -l merged_csv/$new_file.csv)
		echo "================================================="

done

