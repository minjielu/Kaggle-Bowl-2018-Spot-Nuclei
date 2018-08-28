import csv

combined_result = open('final_result.csv','w')
str_1 = './Partial_results/result_'
for x in range(1,12):
    filename = str_1+str(x)+'.csv'
    partial_result = open(filename,'r')
    csvreader = csv.reader(partial_result,delimiter = ',')
    for row in csvreader:
        if x != 1:
            if row[0].startswith('ImageId'):
                continue
        combined_result.write(','.join(row)+'\n')
    partial_result.close()
'''
for x in range(5,12):
    filename = str_1+str(x)+'.csv'
    partial_result = open(filename,'r')
    csvreader = csv.reader(partial_result,delimiter = ',')
    for row in csvreader:
        if x != 1:
            if row[0].startswith('ImageId'):
                continue
        combined_result.write(','.join(row)+'\n')
    partial_result.close()
previous = 'a'
for x in range(4,5):
    filename = str_1+str(x)+'.csv'
    partial_result = open(filename,'r')
    csvreader = csv.reader(partial_result,delimiter = ',')
    for row in csvreader:
        if row[0].startswith('ImageId'):
            continue
        imageid = row[0]
        if imageid == previous:
            continue
        previous = imageid
        combined_result.write(','.join(row)+'\n')
'''
combined_result.close()
