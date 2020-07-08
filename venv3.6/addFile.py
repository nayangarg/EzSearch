import subprocess
import sys

li = sys.argv[1:]

print('Schwartz Hearst')
subprocess.call(["python3", "schwartz_hearst.py", li[0]])
print('Create Graph')
subprocess.call(["python2", "/home/nayan/coding/Major/EzSearch/venv/Create-graph-sCAKE.py", li[0]])
print('Influence Evaluation')
subprocess.call(["python2", "/home/nayan/coding/Major/EzSearch/venv/InfluenceEvaluation.py", li[0]])
print('Word Scoring')
subprocess.call(["python2", "/home/nayan/coding/Major/EzSearch/venv/Word-score-with-PositionWeight-sCAKE.py", li[0]])
print('Updating wordsDocs')
subprocess.call(["python3", "/home/nayan/coding/Major/EzSearch/venv3.6/word2Doc.py", li[0]])
	# os.system("")