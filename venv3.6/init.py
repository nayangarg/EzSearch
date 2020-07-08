import subprocess
import sys
import mysql.connector as conn

print("Enter Path to Parent Folder of Data Folder")
path = input()

mydb = conn.connect(host='localhost', user='EasySearch', passwd = '123', database = 'es7', buffered = True) 


subprocess.call(["python3", "schwartz_hearst.py", path)
# print('Create Graph')
subprocess.call(["python2", path + "/Create-graph-sCAKE.py", path)
# print('Influence Evaluation')
subprocess.call(["python2", path + "/InfluenceEvaluation.py", path)
# print('Word Scoring')
subprocess.call(["python2", path + "/Word-score-with-PositionWeight-sCAKE.py", path)
subprocess.call(["python2", "corpusGlove.py", path)
subprocess.call([path[:-5] + "/glove/demo.sh"])

cur = mydb.cursor(buffered = True)

cur.execute('CREATE TABLE wordSim(Word1 varchar(100), Word2 varchar(100), Distance float)')
cur.execute('CREATE TABLE wordsDocs(Word varchar(100), Doc varchar(100), Rank int)')

subprocess.call(["python3", "wordSim.py")
# print('Updating wordsDocs')
subprocess.call(["python3", "word2Doc.py", path)