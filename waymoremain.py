# -*- coding: utf-8 -*-
"""
Created on Fri Jun 23 10:50:07 2017

@author: Marie
"""
import copy
import collections
from pathlib import Path
import pickle
import random
import math
import numpy as np
#np.set_printoptions(threshold=np.nan)
from sklearn.manifold import MDS
from sklearn import svm
import matplotlib.pyplot as plt
import os

path = ""


mstitlesfile= "dataset/merged_selected_titles.txt"
msreviewsfile2 = "dataset/merged_selected_reviews_nostopwords.txt"


def get_files(reviewsfile):
    with open(reviewsfile, encoding="utf-8", mode="r") as file:
        reviews = [preparestring(line) for line in file]
        
    with open(mstitlesfile, encoding="utf-8", mode="r") as file:
        titles = [line for line in file]        

    return reviews, titles


def preparestring(string):
    str = copy.deepcopy(string)
    str = str.lower()
    str = str.replace(",", "")
    str = str.replace(":", "")
    str = str.replace("(", "")
    str = str.replace(")", "")
    str = str.replace("...", "")
    str = str.replace(".", "")
    str = str.replace(";", "")
    str = str.replace('"', "")
    str = str.replace("?", "")
    str = str.replace("!", "")
    str = str.replace("-", "")
    str = str.replace("???", "")
    str = str.replace("!!!", "")
    str = str.replace("``", "")
    str = str.replace("''", "")
    str = str.replace("'", "")
    str = str.replace("'m", " am")
    str = str.replace("n't", "not")
    str = str.replace("'d", " would")
    str = str.replace("'ve", " have")
    str = str.replace("w/o", "without")
    str = str.replace("w/", "with")
    str = str.replace("'ll", " will")
    str = str.replace("'s", "")
    
    while str.find("  ") > 0: str = str.replace("  "," ")
    if str.endswith(' '): str = str[:-1]
    return str



class thedataset(object):
    def __init__(self, reviews, names, lookup, uplook, count):
        self.reviews = reviews
        self.names = names
        self.lookup = lookup
        self.uplook = uplook
        self.ohnum = count+1  #len(lookup)

               
    def prepareback(self, str):
        str = str.replace(" <comma>", ",")
        str = str.replace(" <colon>", ":")
        str = str.replace(" <openBracket>", "(")
        str = str.replace(" <closeBracket>", ")")
        str = str.replace(" <dots>", "...")
        str = str.replace(" <dot>", ".")
        str = str.replace(" <semicolon>", ";")
        str = str.replace("<quote>", '"')
        str = str.replace(" <question>", "?")
        str = str.replace(" <exclamation>", "!")
        str = str.replace(" <hyphen> ","-")
        str = str.replace(" <END>", "")
        str = str.replace(" <SuperQuestion>", "???")
        str = str.replace(" <SuperExclamation>", "!!!")
        return str
    



def make_dataset(reviews, titles):
    allwords = {}
    wordcount = 2
    
    #first we look how often each word occurs, to delete single occurences.
    string = []
    for line in reviews: 
        words = line.split()
        for word in words:
            string.append(word)
   
    #now we delete single occurences.
    count = []
    count2 = []
    count.extend(collections.Counter(string).most_common(999999999))
    for elem in count:
        if elem[1] > 1:
            count2.append(elem[0])
        
    print("Most common words:")
    print(count[0:5])

    
    #now we make a dictionary, mapping words to their indices
    for line in reviews:
        words = line.split()
        for word in words:
            if not word in allwords:
                if word in count2: #words that only occur once don't count.
                    allwords[word] = wordcount
                    wordcount = wordcount +1
                    
    #print(allwords)            
    
    #the token for single occurences is "<UNK>", the one for end of sentence (if needed) is reserved to 1
    allwords["<UNK>"] = 0
    allwords["<EOS>"] = 1            
    reverse_dictionary = dict(zip(allwords.values(), allwords.keys()))
        
    #now we make every ratings-string to an array of the respective numbers.
    ratings = []     
    for line in reviews:
        words = line.split()
        currentrating = []
        if len(words) > 1:
            for word in words:
                try:
                    currentrating.append(allwords[word])
                except KeyError:
                    currentrating.append(allwords["<UNK>"])
            ratings.append(currentrating)   
            
            
            
    #we made a dataset! :)
    datset = thedataset(ratings, titles, allwords, reverse_dictionary, wordcount)
    
    #add number occurences
    string = []
    for line in dataset.reviews: 
        for word in line:
            string.append(word)
   
    count = collections.Counter(string).most_common(999999999)
    #print([dataset.uplook[i[0]] for i in count[:20]])
    dataset.numoccurences = dict(count)    
    
       
    
    #add the ppmis for the data
    ppmirep = []
    allwordsinalltexts = np.sum([len(i) for i in datset.reviews]) #bei der rechnung von p_et ist der nenner diese konstante.
    for number in range(len(datset.reviews)):
        occurences = dict(collections.Counter(datset.reviews[number]).most_common(999999999))
        ppmis = {}
        pe = np.sum(len(datset.reviews[number])/allwordsinalltexts) #anzahl wörter in diesem text/anzahl wörter insgesamt
        for currword in occurences.keys():
            pet = occurences[currword]/allwordsinalltexts #wie oft kommt DIESES wort in DIESEM text vor
            pt = datset.numoccurences[currword]/allwordsinalltexts #wie oft kommt DIESES WORT insgesamt wor (normiert)
            pmi = math.log(pet/(pe*pt))
            ppmi = max(0,pmi)
            ppmis[currword] = ppmi
    #    print(list(zip([datset.uplook[i] for i in datset.reviews[number][0:100]], [ppmis[i] for i in datset.reviews[number][0:100]])))
        ppmirep.append(ppmis)    
    
    datset.ppmis = ppmirep     
    
    return datset




def load_dataset():    
    print('Loading data...')
    
    if Path(path+"dataset.pkl").is_file():
        print("Dataset found!")
        with open(path+"dataset.pkl", 'rb') as input:
            datset = pickle.load(input)       
    else:
        datset = make_dataset(*get_files(msreviewsfile2))
        
        with open(path+"dataset.pkl", 'wb') as output:
            pickle.dump(datset, output, pickle.HIGHEST_PROTOCOL)
            print('Saved the dataset as Pickle-File')
        
        print(""+str(datset.ohnum)+" different words.")
        rand = round(random.uniform(0,len(datset.names)))
        print('Sample string', datset.reviews[rand][0:100], [datset.uplook[i] for i in datset.reviews[rand][0:100]])
        
           
    print('Data loaded.')
    return datset




def get_ppmidata(dataset):
#    lookat = 3; print([(dataset.uplook[i],dataset.ppmis[lookat][i]) for i in dataset.reviews[lookat][0:100]]) #show a reviewset and their ppmi values
    
#    #look how sparse it is!
#    whichreview = 3                   
#    multidim = []
#    for i in range(dataset.ohnum):
#        try:
#            multidim.append(dataset.ppmis[whichreview][i])
#        except:
#            multidim.append(0)
#    
#    print(multidim[:100])

    if Path(path+"dataset_ppmi.pkl").is_file():
        with open(path+"dataset_ppmi.pkl", 'rb') as input:
            ppmsreviews = pickle.load(input)      
            print("PPMI-stuff loaded!")
    else:
    
        ppmsreviews = []
        for whichreview in range(len(dataset.names)):   
            
            if whichreview % 100 == 0:
                print("Iteration",whichreview,"from",len(dataset.names))
            
            curr = []
            for i in range(dataset.ohnum):
                try:
                    curr.append(dataset.ppmis[whichreview][i])
                except:
                    curr.append(0)
            ppmsreviews.append(curr)
        
        with open(path+"dataset_ppmi.pkl", 'wb') as output:
            pickle.dump(ppmsreviews, output, pickle.HIGHEST_PROTOCOL)
            print('Saved the ppmistuff as Pickle-File')

    return ppmsreviews



def create_distancemat(dataset, ppms):
     if Path(path+"dataset_distancemat.pkl").is_file():
        with open(path+"dataset_distancemat.pkl", 'rb') as input:
            distancematrix = pickle.load(input)      
            print("Distancemat-stuff loaded!")
     else:
        num = len(dataset.reviews)
    
        distancematrix = np.zeros([num,num])
        
        for first in range(num): #pairwise distance is (x²-x)/2 items
            for second in range(first+1, num):
                distancematrix[first,second] = distancematrix[second,first] = abs(np.linalg.norm(ppms[first]-ppms[second]))
            
            if first % 100 == 0:
                print("Iteration",first,"from",num)
        
        with open(path+"dataset_distancemat.pkl", 'wb') as output:
                pickle.dump(distancematrix, output, pickle.HIGHEST_PROTOCOL)
                print('Saved the distancemat as Pickle-File')

     return distancematrix



def make_MDS(dataset, ppms, n_components):
    
    if Path(path+"mds_"+str(n_components)+"D.pkl").is_file():
        with open(path+"mds_"+str(n_components)+"D.pkl", 'rb') as input:
            mdstransform = pickle.load(input)      
            print("MDS-Transform-stuff loaded!")
    else:
        distancematrix = create_distancemat(dataset, ppms)
        
    #    print(distancematrix)
        
        distancematrix = np.triu(distancematrix) + np.transpose(np.triu(distancematrix)) #sonst hat er rundungsfehler >.<
        
        mds = MDS(n_components=n_components)
        
        mdstransform = mds.fit_transform(distancematrix)
        
        with open(path+"mds_"+str(n_components)+"D.pkl", 'wb') as output:
            pickle.dump(mdstransform, output, pickle.HIGHEST_PROTOCOL)
            print('Saved the MDS-Transform as Pickle-File')
    
    return mdstransform
    

def return_elem(movietitle):
    with open(mstitlesfile, encoding="utf-8", mode="r") as file:
        counter = 0
        for line in file:
            if line.replace("\n","") == movietitle:
                break
            counter += 1
        
    return counter




def print_distance(movie1, movie2, dimensionrepresentation):
    m1 = return_elem(movie1)
    m2 = return_elem(movie2)
    print("Distance between",movie1,"and",movie2,":",abs(np.linalg.norm(dimensionrepresentation[m1]-dimensionrepresentation[m2])))


def find_termcandidates(dataset):
    prettyoften = dict((k, v) for k, v in dataset.numoccurences.items() if 100 <= v <= 10000)
    print([dataset.uplook[i] for i in list(prettyoften.keys())[:10]])
    candidates = dict(zip(list(prettyoften.keys()),np.zeros(len(list(prettyoften.keys())))))  
    for i in dataset.reviews:
        for j in candidates.keys():
            if j in i:
                candidates[j] += 1

    print(len(candidates))
    prettyoften = dict((k, v) for k, v in candidates.items() if 100 <= v <= 1500)
    print(len(prettyoften))
    print([dataset.uplook[i] for i in list(prettyoften.keys())[:10]])                
    #prints ['comedy', 'series', 'girl', 'american', 'beautiful', 'book', 'effects', 'family', 'black', 'dead']                
    
    dataset.termcandidates = prettyoften
    
    with open(path+"dataset.pkl", 'wb') as output:
        pickle.dump(dataset, output, pickle.HIGHEST_PROTOCOL)
        print('Saved the dataset as Pickle-File')    


def show_some_stuff(dataset, ppms, mdstrans):
    print([dataset.uplook[i] for i in dataset.reviews[0][0:100]])
    print(ppms[3][:100])

    mdstrans = np.array(mdstrans)
    print(mdstrans.shape)
    print_distance("The Shining", "Saw", mdstrans)
    print_distance("The Shining", "Herbie Fully Loaded", mdstrans)
    print_distance("Herbie Fully Loaded", "Saw", mdstrans)
    print_distance("Star Trek V: The Final Frontier", "&#x22;Star Trek: Hidden Frontier&#x22;", mdstrans)
    print_distance("Star Trek V: The Final Frontier", "The Shining", mdstrans)
    print_distance("Star Trek V: The Final Frontier", "Saw", mdstrans)    




def run_svm(dataset, PREmdstrans, words, num_dims, shorten=0):
    if not type(words) is list: 
        words = [words]
        
    if shorten == 0:
        shorten = len(dataset.reviews)
        
    if os.path.exists(path+"supportvectormachine_"+"_".join(words)+"_"+str(num_dims)+"D.pkl"):
        with open(path+"supportvectormachine_"+"_".join(words)+"_"+str(num_dims)+"D.pkl", 'rb') as input:
            clf = pickle.load(input)      
            print("supportvectormachine loaded!")    
            
    else:
        for word in words:
            print(word,"occurs in",dataset.termcandidates[dataset.lookup[word]],"different reviews")
            
        reviews = dataset.reviews[:shorten]
        mdstrans = PREmdstrans[:shorten]
        
        kommtvor = [False]*len(reviews)
        for word in words:
            for i in range(len(reviews)):
                if dataset.lookup[word] in reviews[i]:
                    kommtvor[i] = True
    
        tmp = [str(i[0])+" "+i[1] for i in list(zip(kommtvor, dataset.names))]
        with open(path+"positives_"+"_".join(words)+".txt", encoding="utf-8", mode="w") as outfile:
            outfile.write("".join(tmp))
    
            
        clf = svm.SVC(kernel='linear', class_weight={1: (len(kommtvor)/np.sum(kommtvor)*0.5)})
        clf.fit(mdstrans, kommtvor)
        
        with open(path+"supportvectormachine_"+"_".join(words)+"_"+str(num_dims)+"D.pkl", 'wb') as output:
            pickle.dump(clf, output, pickle.HIGHEST_PROTOCOL)
            print('Saved the supportvectormachine as Pickle-File')


    kommtvor = [False]*len(dataset.reviews[:shorten])
    for word in words:
        for i in range(len(dataset.reviews[:shorten])):
            if dataset.lookup[word] in dataset.reviews[i]:
                kommtvor[i] = True
    


    return clf, kommtvor




if __name__ == '__main__':
    num_dims = 2 #50!
    shorten = 500
    
    dataset = load_dataset()
    ppms = None #ppms = np.array(get_ppmidata(dataset))
    mdstrans = make_MDS(dataset, ppms, num_dims) 
    
    #show_some_stuff(dataset, ppms, mdstrans)
    
    #supportvectormachine = run_svm(dataset, mdstrans, ["comedy","fun"], num_dims, shorten=False)
    #supportvectormachine = run_svm(dataset, mdstrans, "scary", num_dims, shorten=False)
    
    supportvectormachine, kommtvor = run_svm(dataset, mdstrans, ["comedy","hilarious"], num_dims, shorten=shorten)
    
    mdstrans = mdstrans[:shorten]    

    print(supportvectormachine.coef_, supportvectormachine.intercept_)


    if num_dims == 3:
        from mpl_toolkits.mplot3d import Axes3D
        import pylab
        plt.clf()
        fig = pylab.figure()
        ax = Axes3D(fig)
        ax.set_xlim3d(-500, 500)
        ax.set_ylim3d(-500, 500)
        ax.set_zlim3d(-500, 500)
        ax.scatter(xs = mdstrans[:, 0], ys= mdstrans[:, 1], zs= mdstrans[:, 2], c=kommtvor, cmap=plt.cm.Paired)
        plt.show()    
    
    if num_dims == 2:
        #plot supportvectormachine
        w = copy.deepcopy(supportvectormachine.coef_[0]); w[0] = -w[0]
        a = w[0] / w[1]
        #xx = np.linspace(min([i[0] for i in mdstrans]), max([i[0] for i in mdstrans]))
        xx = np.linspace(-300,500)
        yy = a * xx - (supportvectormachine.intercept_[0]) / w[1]
        plt.plot(xx, yy, 'k-')
        
        b = -w[1] / w[0]
        yy2 = b * xx
        plt.plot(xx, yy2, c="Red")
        
        # plot the points, and the nearest vectors to the plane
        plt.scatter(mdstrans[:, 0], mdstrans[:, 1], c=kommtvor, cmap=plt.cm.Paired)
    
        plt.gca().set_aspect('equal', adjustable='box')
        plt.axis([min(xx.extend(yy)),max(xx.extend(yy)),min(xx.extend(yy)),max(xx.extend(yy))])
        plt.show()    
    

