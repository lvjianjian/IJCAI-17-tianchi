startid=1601
range=20
endid=2000
while (($startid < $endid)) 
	do
		echo $startid $((startid+range-1))
		python RNN/SRN.py $startid $((startid+range-1))
		startid=$((startid+range))
	done
