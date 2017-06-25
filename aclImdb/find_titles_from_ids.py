# -*- coding: utf-8 -*-
"""
Created on Wed Jun 21 16:57:40 2017

@author: csten_000
"""
import re
import urllib.request

basepath = ""

def create_titlenames():
    for nextpiece in ["test/", "train/"]:
        path = basepath + nextpiece
        for lastpiece in ["neg", "pos", "unsup"]:
    
            if nextpiece == "test/" and lastpiece == "unsup": 
                continue
            
            string = []
            try:
                with open(path+"titles_"+lastpiece+".txt", encoding="utf8") as f:
                    for i, l in enumerate(f):
                        pass
                otherfilelen = i
                string.append("")
            except:
                otherfilelen = 0
                
            
            try:
                with open(path+"urls_"+lastpiece+".txt", encoding="utf8") as infile:
                    counter = 0
                    lastlink = ""
                    for line in infile: 
                        counter = counter+1
                        if otherfilelen > 0 and counter <= otherfilelen+1:
                            continue
                        if line != lastlink:
                            try:
                                imdbid = line[line.find("/title/"):line.find("/usercomments")]
                                with urllib.request.urlopen(line) as response:
                                   html = response.read().decode('utf-8')
                                   try:
                                       lastname = re.findall(r'a class="main" href="'+imdbid+r'/">(.*?)</a>', html)[0]
                                   except IndexError:
                                       newlink = response.geturl()[response.geturl().find("/title/"):response.geturl().find("/reviews")]
                                       lastname = re.findall(r'a class="main" href="'+newlink+r'/">(.*?)</a>', html)[0]
                                   lastlink = line
                            except (urllib.error.HTTPError, urllib.error.URLError, UnicodeDecodeError, ConnectionResetError, TimeoutError):
                                lastname = "NOT_FOUND"
                        print(counter," ",lastname)
                        string.append(lastname)
            finally: #auch bei KeyboardInterrupt!
                dowhat = "a" if otherfilelen > 0 else "w"
                with open(path+"titles_"+lastpiece+".txt", dowhat) as infile:
                    infile.write("\n".join(string))
                    
                    
if __name__ == '__main__':
    create_titlenames()