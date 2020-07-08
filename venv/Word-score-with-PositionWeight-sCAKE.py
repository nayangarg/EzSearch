#3.

import os
import sys
import numpy as np
import pandas as pd
from read_write_create import *

def invert_num(l):
	# prin t(l)
	return [(1.0/x) for x in l if x!=0]

def get_node_weight(w, df_pos):
    
    w_weight = 0
    w_row = df_pos.loc[df_pos["words"] == w]
    
    if len(w_row["words"]) > 0:
        posi = list(w_row["positions"])[0]
        posi = posi[:-1]
        if(len(posi) >0):
            w_weight = sum(invert_num(posi))
    
    return w_weight

#------ main ------#
global path, data_path, word_score_path

# cwd = os.getcwd()
li = sys.argv[1:]

if '/' in li[0]:
    path = li[0]
    data_path = path + "/data"

    create_folder(path,"SCScore_W")
    word_score_path = path + "/SCScore/"

# print("Word-score-with-PositionWeight-sCake")
# li = ['566390.txt']

for every_file in li:#(os.listdir(data_path)):
    
    # print(every_file)
    file_name = every_file[:-4]
    #text = read_text_from_file(data_path,every_file)
    
    f_wordScore = read_text_from_file(path+"/SCScore",file_name+".csv.sortedranked.IF.txt")
    
    with open(path+"/positions/"+file_name+".pkl", 'rb') as f:
        df_pos = pd.read_pickle(f, compression=None)
        
    df = pd.read_csv(word_score_path + file_name  +".csv.sortedranked.IF.txt")
    words = df["Name"]
    wscore = df["IF"]
    
    node_weight = [0] * len(words)
    for index in range(len(words)):
        node_weight[index] = get_node_weight(words[index], df_pos)
    
    new_score = list(wscore * node_weight)
    
    data = dict()
    data["Words"] = words
    data["Old_WScore"] = wscore
    data["Position_Weight"] = node_weight
    data["SCScore"] = new_score
    new_df = pd.DataFrame(data=data)
    new_df = new_df.sort_values("SCScore",ascending=False)
    
    new_df.to_csv(path+"/SCScore_W/"+ file_name +"_ranked_list.csv")
     
