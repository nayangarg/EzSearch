import mysql.connector as conn
import sys

mydb = conn.connect(host='localhost', user='EasySearch', passwd = '123', database = 'es7', buffered = True)

cur = mydb.cursor(buffered = True)

# path = '/home/nayan/coding/Major/EasySearch/venv/SCScore_W'
# data_path = '/home/nayan/coding/Major/EasySearch/venv/data'
import os
import pandas as pd
import csv

n = 5

li = sys.argv[1:]

if '/' in li:
    path = li[0]
    data_path = path + '/data'
    li = os.listdir(path)
else:
    li[0] = li[0][:-4] + '_ranked_list.csv'

for every_file in li:#(os.listdir(path)):
    # print(every_file)
    df = pd.read_csv(open(path+'/'+every_file, 'r'))
#     print(df.head())
    l = df['Words'].size
    # print(l)
    
    for i in range(l):
        # print(i)
        if pd.isnull(df['Words'][i]):
            continue
        cur.execute("select * from wordsDocs where Word = %s and Doc = %s", (df['Words'][i],data_path+'/'+every_file[:-16]+'.txt'))
        a = cur.fetchone()
        # print(a)
        if (a):
            # print('saf')
            continue

        cur.execute("select count(*) from wordsDocs where Word = %s", (df['Words'][i],))
        a = cur.fetchone()
        # print(df['Words'][i], a)
        if not a or a[0] < n:
            # print(df['Words'][i], 'insert')
            cur.execute('insert into wordsDocs values(%s,%s,%s)',(df['Words'][i], data_path+'/'+every_file[:-16]+'.txt', i))
        else:
            cur.execute("select max(Rank) from wordsDocs where Word = %s group by Word", (df['Words'][i],))
            b = cur.fetchone()[0]
            # print(b)
            if b > i:
                # print(df['Words'][i],'update')
                cur.execute('update wordsDocs set Doc = %s, Rank = %s where Word = %s and Rank = %s limit 1', (data_path+'/'+every_file[:-16]+'.txt', i, df['Words'][i], b))     
        mydb.commit()    