import re
import nltk
from nltk.stem import PorterStemmer
import csv
from autocorrect import Speller
from nltk.corpus import *
from read_write_create import *
import time
import atexit


dic = {}

class Node:
	
	def __init__(self, key, data):
		self.key = key
		self.data = data
		self.next = None
		self.prev = None

class DoublyLL:

	def __init__(self, mxSiz):
		self.mxSiz = mxSiz
		self.crtSiz = 0
		self.head = None
		self.tail = None

	def pushToHead(self,x):
		if x.prev:
			x.prev.next = x.next
			x.next = self.head
			self.head = x

	def overflow(self):
		# print('vs')
		# print(self.tail.key)
		del dic[self.tail.key]
		# print(self.tail.key)
		self.tail = self.tail.prev
		if self.tail:
			self.tail.next = None
		# print(self.tail.key)
		self.crtSiz -= 1

	def addNew(self, key, data):
		x = Node(key, data)
		if self.head:
			x.next = self.head
			self.head.prev = x
		self.head = x
		# print(self.crtSiz)
		self.crtSiz += 1
		# print(self.crtSiz)

		if self.crtSiz == 1:
			self.tail = self.head

		dic[key] = x		

ll = DoublyLL(10);


def search(text):
	# print(ll.crtSiz)
	if text in dic.keys():

		x = dic[text]
		ll.pushToHead(x)

		return x.data

	else:
		res = []
		qry = []

		newline_ex = re.compile("\n")
		numbers_ex = re.compile("[0-9]+(-[0-9]+)?")
		punctuation_ex = re.compile("[^a-z]-")
		roman_num_ex = re.compile("\\b[i|v|x|l|c|d|m]{1,3}\\b")
		stopwords = read_list_from_file('/home/nayan/coding/Major/EzSearch/venv', "stopwords.txt")
		ps = PorterStemmer()
		check = Speller(lang='en')

		text = re.sub(newline_ex, ' ', text)
		# text = unicode(text, errors='replace')
		#sen = convert_text_to_sentences(text)

		## pre-processing text
		text = text.strip()
		text = text.lower()
		text = re.sub(numbers_ex, '', text)
		text = re.sub(punctuation_ex, '', text)
		text = re.sub(roman_num_ex, '', text)
		text= text.replace("-", " ")

		words = nltk.word_tokenize(text)

		# Spell-Check

		words =[check(i) for i in words]
		# print(words)

		# StopWords

		words = [i for i in words if i not in stopwords]
		# print(words)
		            
		# qry = list(set(qry))
		# print(qry)

		# Stemming

		words = [ps.stem(i) for i in words]

		l = len(words)
		if (l > 3):
			qry.append('-'.join(words))

		# print(qry)
		# Bi-Tri Grams

		for i in range(l-2):
		    qry.append(words[i]+'-'+words[i+1]+'-'+words[i+2])

		for i in range(l-1):
		    qry.append(words[i]+'-'+words[i+1])

		for i in words:
		    qry.append(i)

		# Synonyms
		syno = set()
		for i in words:
		    for syn in wordnet.synsets(i): 
		      for l in syn.lemmas():
		        j = l.name()
		#         print(j)
		        j = j.replace('_', '-')
		#         print(j)
		        syno.add(ps.stem(j)) 

		# qry = list(set(qry))
		# print(qry)

		import mysql.connector as conn
		mydb = conn.connect(host='localhost', user='EasySearch', passwd = '123', database = 'es7', buffered = True)
		cur = mydb.cursor(buffered = True)
		simWords = list()
		docs = set()

		f=0

		res.append('')

		for i in qry:
			cur.execute('select Word2 from wordSim where Word1 = %s',(i,))
			a = cur.fetchall()
			mydb.commit()
			cur.execute('select Doc from wordsDocs where Word = %s',(i,))
			b = cur.fetchall()
			if not b:
				continue
			f=1
			res.append(i)
			for d in b:
				# if d[0] not in docs:
				res.append(d[0])
				# docs.add(d[0])
			res.append('')
			#     print(cur.fetchall())
			mydb.commit()
			for j in a:
				# print(j[0])
				if j[0] not in qry and j[0] not in simWords:
					
					simWords.append(j[0])
					# cur.execute('select Doc from wordsDocs where Word = %s',(j[0],))
					# b = cur.fetchall()
					# if not b:
					# 	continue
					# print(j[0])
					# for d in b:
					# 	# if d[0] not in docs:
					# 	print(d[0])
					# 	# docs.add(d[0])	
					# print('')
					# #             print(cur.fetchall())
					# mydb.commit()
				# mydb.commit()

		for i in simWords:
			# if i not in qry:
			cur.execute('select Doc from wordsDocs where Word = %s',(i,))
			b = cur.fetchall()
			if not b:
				continue
			f=1
			res.append(i)
			for d in b:
				# if d[0] not in docs:
				res.append(d[0])
				# docs.add(d[0])
			res.append('')
			mydb.commit()

		for i in syno:
			if i in qry or i in simWords:
				continue
			cur.execute('select Doc from wordsDocs where Word = %s',(i,))
			b = cur.fetchall()
			if not b:
				continue
			f=1
			res.append(i)
			for d in b:
				# if d[0] not in docs:
				res.append(d[0])
				# docs.add(d[0])
			res.append('')
			mydb.commit()

		if f==0:
			res.append('No Results Found')

		if ll.crtSiz == ll.mxSiz:
			ll.overflow()
		# print(ll.crtSiz)
		ll.addNew(text, res)
		return res

def onExit():
	wrtr = csv.writer(open('ExportedLL.csv', 'w'))
	while ll.tail:
		wrtr.writerow(ll.tail.key)
		wrtr.writerow(ll.tail.data)
		ll.tail = ll.tail.prev

atexit.register(onExit)

rdr = list(csv.reader(open('ExportedLL.csv', 'r')))
l = len(rdr)
k = ''

for i in range(l):
	if (i%2==0):		
		for j in rdr[i]:
			k += j
		
	else:
		dat = []
		for j in rdr[i]:
			dat.append(j)
		ll.addNew(k,dat)
		k = ''

while (1):
	# print(crtSiz)
	
	text = input("Search: ")
	t = time.time()
	# time.sleep(10)
	
	# if (text == 'quit'):
	# 	sm_pid = os.getpid()
	# 	print(sm_pid)
	# 	p = psutil.Process(sm_pid)
	# 	p.suspend()
	# 	# p.resume()
	# # os.system('fg')
	# else:
	# print(ll.crtSiz)
	res = search(text)

	# l = len(res)

	for i in res:
		print(i)
	# print(ll.crtSiz, ll.head.key, ll.tail.key)
		
	print('Query Time: ', time.time()-t)

