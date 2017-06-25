# -*- coding: utf-8 -*-
"""
Created on Thu Jun 22 22:11:03 2017

@author: Marie
"""

import numpy as np
import re

titlesfile= "dataset/all_titles.txt"
reviewsfile = "dataset/all_reviews.txt"

mtitlesfile= "dataset/merged_titles.txt"
mreviewsfile = "dataset/merged_reviews.txt"

mstitlesfile= "dataset/merged_selected_titles.txt"
msreviewsfile = "dataset/merged_selected_reviews.txt"
msreviewsfile2 = "dataset/merged_selected_reviews_nostopwords.txt"

stopwordsfile = "stopwords.txt"

def find_movie():

    titlepart = input("Some Part of a Movietitle...: ").lower()
    candidates = []
    
    with open(titlesfile, encoding="utf-8", mode="r") as file:
        for line in file:
            if titlepart in line.lower():
                if not line.replace("\n","") in candidates:
                    candidates.append(line.replace("\n",""))
                    
    print("Possible movies (select with number):")
    
    candistring = ""
    
    for i in range(len(candidates)):
        candistring += "("+str(i)+")  "+candidates[i]+"\n"
    candistring = candistring[:-1]
    
    print(candistring)
    
    selection = input("Which one did you mean? ")
    
    selectedmovie = candidates[int(selection)]
    print("You selected: "+selectedmovie)

    return selectedmovie


def return_reviews(movietitle):
    lines = []
    with open(titlesfile, encoding="utf-8", mode="r") as file:
        counter = 0
        for line in file:
            if line.replace("\n","") == movietitle:
                lines.append(counter)
            counter += 1
            
    reviews = []
    with open(reviewsfile, encoding="utf-8", mode="r") as file:
        counter = 0
        for line in file:
            if counter in lines:
                reviews.append(line.replace("\n",""))
            counter += 1        
    
    return reviews
            
        
def merge_reviews():
    considered = []
    appending = False
    try:
        with open(mtitlesfile, encoding="utf-8", mode="r") as file:
            for line in file:
                text = line.replace("\n","")        
                considered.append(text)
                appending = True
    except:
        pass
    
    print(len(considered))
    
    allreviews = []
    alltitles = []
    counter = 0
    with open(titlesfile, encoding="utf-8", mode="r") as file:
        for line in file:
            text = line.replace("\n","")
            
            if not text in considered:
                allreviews.append(" ".join(return_reviews(text)))
                considered.append(text)
                alltitles.append(text)
                
            if counter % 250 == 0 and counter>0:
                print("Iteration:",counter)
                            
                with open(mreviewsfile, encoding="utf-8", mode="a") as outfile:
                    if appending: 
                        outfile.write("\n")
                    outfile.write("\n".join(allreviews))
                    
                with open(mtitlesfile, encoding="utf-8", mode="a") as outfile:
                    if appending: 
                        outfile.write("\n")
                    outfile.write("\n".join(alltitles))
                
                allreviews = []
                alltitles = []
                
            counter += 1
    print(len(considered))




def add_to_one_file():
    alllens = []
    with open(mreviewsfile, encoding="utf-8", mode="r") as file:
        for line in file:
            alllens.append(len(line.split(" ")))
            
    alllens = np.array(alllens)
    mean = round(np.mean(alllens))
    
    takeit = [True if i >= 0.5*mean else False for i in alllens]
    
    print(takeit)
    
    msreviews = []
    with open(mreviewsfile, encoding="utf-8", mode="r") as file:
        counter = 0
        for line in file:
            if takeit[counter]:
                msreviews.append(line.replace("\n",""))
            counter += 1

    mstitles = []    
    with open(mtitlesfile, encoding="utf-8", mode="r") as file:
        counter = 0
        for line in file:
            if takeit[counter]:
                mstitles.append(line.replace("\n",""))
            counter += 1
    
    
    with open(msreviewsfile, encoding="utf-8", mode="w") as outfile:
        outfile.write("\n".join(msreviews))
    
    with open(mstitlesfile, encoding="utf-8", mode="w") as outfile:
        outfile.write("\n".join(mstitles))



def get_files(reviewsfile):
    with open(reviewsfile, encoding="utf-8", mode="r") as file:
        reviews = [line for line in file]
        
    with open(mstitlesfile, encoding="utf-8", mode="r") as file:
        titles = [line for line in file]        

    return reviews, titles


def get_stopwords():
    stopwords = []
    with open(stopwordsfile, encoding="utf-8", mode="r") as file:
        for line in file:
            line = line.replace(" ","").replace("\n","")
            if line.find("|") > 0: line = line[:line.find("|")]
            if line != "" and line[0] != "|":
                stopwords.append(line)
    return stopwords

def remove_stopwords():
    reviews, titles = get_files(msreviewsfile)
    stopwords = get_stopwords()
    newreviews = []
    for curr in reviews:
        for word in stopwords:
            curr = re.sub(r"\b"+word+r"\b","",curr)
        new = curr.replace("\n"," ")
        new = re.sub(' +',' ',new)
        if new[0] == " ": 
            new = new[1:]
        newreviews.append(new)
        
    with open(msreviewsfile2, encoding="utf-8", mode="w") as outfile:
        outfile.write("\n".join(newreviews))



if __name__ == '__main__':
    reviews, titles = get_files(msreviewsfile2)

                
    
    
        
#    merge_reviews()
    #movie1 = find_movie()
#    movie1 = "The Shining"
#    print(len(return_reviews(movie1)))

