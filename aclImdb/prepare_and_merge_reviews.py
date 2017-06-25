# -*- coding: utf-8 -*-
"""
Created on Thu Jun 22 15:34:01 2017

@author: csten_000
"""

import fnmatch
import os
import nltk


for path in ["test/neg/", "test/pos/", "train/neg/", "train/pos/", "train/unsup/"]:
    
    reviews = []
    
    try:
        with open(path+"00_allreviews.txt", encoding="latin-1") as f:
            for i, l in enumerate(f):
                pass
        otherfilelen = i
        reviews.append("")
    except FileNotFoundError:
        otherfilelen = 0    
        
        
    for i in range(100000):    
    
        if otherfilelen > 0 and i <= otherfilelen:
            continue    
    
            
        #get the sentence from file
        try:
            for file in os.listdir(path):
                if fnmatch.fnmatch(file, str(i)+'_*.txt'):
                    raise StopIteration
            break #wenn er das nicht findet kommt er zu diesem break und breakt aus dem main for loop
        except StopIteration:
            with open(path+file, encoding="latin-1") as infile:
                string = " ".join([line for line in infile])
                
        newstring = string.lower()
        newstring = newstring.replace("<br />"," ")
        sentences = nltk.sent_tokenize(newstring)
        
        sent = []
        
        for sentence in sentences:
            
            tokens = nltk.word_tokenize(sentence)
            tagged = nltk.pos_tag(tokens)
                    
            chunked = nltk.ne_chunk(tagged)
            
            tagged2 = nltk.pos_tag([i[0] if isinstance(i[0], str) else "_".join([j[0] for j in i]) for i in chunked])
            
            grammar = "NP: {<NN.*>+}" #"NP: { <DT>?<JJ.*>*<NN.*>+}"
            
            cp = nltk.RegexpParser(grammar)
            result = cp.parse(tagged2)
            sent.extend([i[0] if isinstance(i[0], str) else "_".join([j[0] for j in i]) for i in result])
        
        curreview = " ".join(sent)
        reviews.append(curreview)
        
        
        if (i+1) % 100 == 0:
            print("Iteration:",i+1)
                
            #print("\n".join(reviews))
            with open(path+"00_allreviews.txt", "a", encoding="utf-8") as outfile:
                outfile.write("\n".join(reviews).encode('utf-8','ignore').decode('utf-8'))
            
            reviews = [""]