# EzSearch

A search engine for the local filesystem dedicated to the Research community.

"EasySearch" is a push to straightforwardness and advances the procedure of local search, for
example off the web search on a person's PC or laptop. Specially aimed at research scientists who
have a humongous number of research papers saved offline; a great deal of manual work is required
to fetch documents holding words or phrases that are related to a certain topic. A simple example
could be papers related to machine learning. While scanning for this expression, it would be of
colossal assistance if papers identified with subjects like Neural Nets, Artificial Intelligence and so
forth likewise turn up. This is the distinctive feature of our project, apart from exact word matching
our model is robust in displaying semantically similar words to a user’s query.
The aim- “a semantic search” is kept in mind from the very beginning as the first process of this
task itself is semantically aware. We utilize a parameter less method for keyphrase extraction by
constructing a graph of text that captures the contextual relation between words. A word scoring
strategy is likewise utilized dependent on the association between themes or concepts. Once
keywords and keyphrases are extracted from every document, a connection between all the
extracted words is made with the help of word embeddings. The entries are fed into a database of
words. A user can look for a specific subject by entering a query, the input query is pre-processed
and the results are fetched from the database. The corresponding term and the terms that are closest
to it in the database are displayed as the output along with metadata of related documents.

#User Manual

    1. Download and install Python environment (Version 2.7 and 3.6).
    2. Install Mysql.
    3. Save the files in a folder.
    4. Go to the folder glove.
        a. Edit the path to corpus.txt, vocab.txt, and vectors.txt in demo.sh (~/venv3.6/~). 
    5. Create new User in Mysql.
        a. Username : EasySearch
        b. Password : 123
        c. You can change the password, as you please.
    6. Run ~/venv3.6/init.py.
    7. The model will take some time to build.
    8. Once the model is built, the software is ready to use. 
        a. Open Terminal. 
        b. Change to the directory (~/venv3.6). 
        c. Type “python3 qrySrch.py”.
        d. Press Enter.
    9. Search the required queries.
    10.  In order to add a new file.
        a. Move file to data folder.
        b. Open Terminal. 
        c. Change to the directory (~/venv3.6). 
        d. Type “python3 addFile.py <NewFile>”.
        e. Press Enter.
    11.  In order to update the Glove Model and Word Similarities.
        a.  Open Terminal. 
        b. Change to the directory (~/venv3.6). 
        c. Type “python3 updateWordSim.py”.
        d. Press Enter.
