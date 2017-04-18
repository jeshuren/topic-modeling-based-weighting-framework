#!/bin/bash

# This script is used for computing svm distances for every CV split.

# use the base file name here.
filename='letterj.csvtrain'
csv=".csv"
echo $filename

# run across 4 Cross validation rounds
for j in {1..4}
do
	# run through 5 folds of every cross validation round.
	for i in {1..5};
	do
		echo $filename
		in="$filename$j$i$csv"
		out="$filename$j$i"
		echo $in
		echo $out
		
		# run the svm distance computation routine.
		python svm_distance.py -t 0 -c 2 -o $out $in
	done
done
