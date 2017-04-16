#!/bin/bash

filename='letterj.csvtrain'
csv=".csv"
echo $filename
for j in {1..4}
do
	for i in {1..5};
	do
		echo $filename
		in="$filename$j$i$csv"
		out="$filename$j$i"
		echo $in
		echo $out
		python svm_distance.py -t 0 -c 2 -o $out $in
	done
done
