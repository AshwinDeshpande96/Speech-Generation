import os
import numpy as np

i = 1
president_list = [str(p) for p in os.listdir(os.getcwd()) if os.path.isdir(p)]
for president in president_list: 
    if os.path.isdir(president):
        print i, "\b) ",  president, "\b's ", len(os.listdir(president)), "speeches"
        i = i + 1
choice = input("Choose President: ")

president = president_list[choice-1]
print "President: "+president

file_count = 0
while file_count<1 or file_count>len(os.listdir(president)):
    file_count = input("Choose number of speeches: ")
output_file_name = president+"_concat_" +str(file_count)+".txt"; 
output_file = open(output_file_name, 'w')

filelist = [president+"/"+file for file in os.listdir(os.getcwd()+"/"+president) if file.endswith(".txt")]

chosen_speeches = list(np.random.choice(filelist, file_count))

char_count = 0
for file in chosen_speeches:
    print file
    fptr = open(file, 'r')
    f_text = fptr.read()
    char_count = char_count + len(f_text)
    f_text = f_text[f_text.index('>')+1:]
    f_text = f_text[f_text.index('>')+1:]
    output_file.write(f_text)
    fptr.close()
    
output_file.close()

print "Concatenated file saved as: "+output_file_name +"at "+ os.path.abspath(output_file_name)
print "Characted count of "+output_file_name +": "+ str(char_count)
