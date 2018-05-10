import re
from itertools import islice

def next_n_lines(file_opened, N):
    return [x.strip() for x in islice(file_opened, N)]

#-# function parameters
RAW_DATASET='dialogues.txt'
FORMATTED_DATASET='dialogues_83097.txt'
NUM_OF_LINES_EXTRACTED=83097

#-# reading raw_data_file
with open(RAW_DATASET, 'r') as file_opened:
    lines = next_n_lines(file_opened, NUM_OF_LINES_EXTRACTED)
    
    
#-# creating new datafile with formatted dataset usable by our program
f=open(FORMATTED_DATASET, 'w')
for line in lines:
    next_line=[]
    next_line=re.split(r'\t+', line)
    for i in range(0,len(next_line)-1):
        add_line= next_line[i]+'\t'+next_line[i+1]+'\n'
        f.write(add_line)
f.close()

#-# number of lines in dataset
num_lines = sum(1 for line in open(FORMATTED_DATASET))
print(num_lines)
