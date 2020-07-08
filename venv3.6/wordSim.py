import mysql.connector as conn
import distance as d
import csv

mydb = conn.connect(host='localhost', user='EasySearch', passwd = '123', database = 'es7', buffered = True)
cur = mydb.cursor(buffered = True)

W, vocab, ivocab = d.generate()

rdr = csv.reader(open('vocab.txt','r'))
rdr = list(rdr)

n = 5
for w in rdr:
    w1 = w[0].split()[0]
    if (len(w1) > 100):
        continue
    b = d.distance(W, vocab, ivocab, w1, n)
    val = []
    
    for i in b:
        w2 = i[0]
        if len(w2) > 100:
            continue
        dist = i[1]
        val.append((w1,w2,float(dist)))
    print(val)
    cur.executemany('insert into wordSim values(%s, %s, %s)', val)
    mydb.commit()
