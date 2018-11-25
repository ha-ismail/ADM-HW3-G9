#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import nltk
import csv

from nltk.corpus import stopwords
stop_words = set(stopwords.words("english"))
from nltk.tokenize import RegexpTokenizer
from nltk.stem.snowball import SnowballStemmer
from translate import Translator
from langdetect import detect

stemmer = SnowballStemmer('english')

from pathlib import Path
import math
import numpy as np
import re
from scipy import spatial
import heapq as hq
import geopy
import geopy.distance
import feather
import deepdish

from IPython.display import HTML, display

# Other functions 

def cleanData(rawData, lang='english'):
        tokenizer = RegexpTokenizer(r'\w+')
        stop_words = set(stopwords.words(lang))

        # get words lowercased
        t0 = rawData.lower()
        # remove puctuations
        t1 = tokenizer.tokenize(t0)

        # reomve stop words
        t2 =[]
        t2 = [t1[i] for i in range(0,len(t1)) if t1[i] not in stop_words]

        # stemm words
        t3 = [stemmer.stem(t2[i]) for i in range(0, len(t2))]

        # remove nummbers and strings starting with numbers
        t4 = [t3[i] for i in range(0, len(t3)) if t3[i][0].isdigit()==False]

        return(t4)
# function to get the cosine similarity between two given vectors

def getCosineSimi(v1,v2):
    return (1 - spatial.distance.cosine(v1, v2))

def saveData(data, file):
    feather.write_dataframe(data, file+'.feather')
        
    

# Data class
class AirbnbData:
    def __init__(self, csvFile, dataName='AirbnbData Texas', buildData = False):
        # initiate class attributes
        self.name = dataName
        self.csvFile = csvFile
        if buildData:
            self.data = pd.read_csv(self.csvFile, usecols= ['average_rate_per_night', 
                                                        'bedrooms_count','city', 
                                                        'date_of_listing', 
                                                        'description', 
                                                        'latitude',
                                                        'longitude',
                                                        'title', 
                                                        'url'])
        self.fileCount = 0
        self.lang = "English"
        self.dataFolder = "data"
        self.vocabularyFile = "vocabulary.csv"
        self.words = []
        self.vocabulary = self.getVocabulary()
        self.voc = self.vocabulary
        # initiate imported modules
 
    # this function can read a dataframe file and assign to Airbnb object if it was initiated without reading data
    def readData(self, file):
        self.data = feather.read_dataframe(file)
    
    # function to show messages when running some functions
    def present(self, outHtml):
        styles = open("style/style.css", "r").read()
        outputHTML = '<style>%s</style>' % styles
        outputHTML += outHtml
        display(HTML(outputHTML))
    
    # cleaning the data
    def clean(self):
        self.data = self.data.dropna(how='any',axis=0, subset=['average_rate_per_night', 
                                                               'bedrooms_count','city', 
                                                               'description', 'latitude',
                                                               'longitude',
                                                               'title'])
        self.fileCount = len(self.data)
        hhtm = '<div class="h_msg">'+self.name+' data has been cleaned from null values sir!</div>'
        
        self.present(hhtm)
    
    def removeDuplicates(self):
        #
        hhtm = '<div class="h_msg">Duplicates data have been removed from '+self.name+' sir!</div>'
        
        self.present(hhtm)
        
    def translate(self,lang="English"):
        # we have noticed some text values in language other than english, so not to lose them we are going to translate them into English 

        translator= Translator(to_lang=lang)
        for index, row in self.data.iterrows():
            try:
                if detect(row['description']) != 'en' and len(row['description']) >3:
                    row['description'] = translator.translate(row['description'])
                if detect(row['title']) != 'en' and len(row['title']) >3:
                    row['title'] = translator.translate(row['title'])
            except :
                print("Error: in detecting language with message")
                print(index)
                self.data.drop([index])
        # reset the index 
        self.data.index = range(len(self.data.index))
        self.fileCount = len(self.data)
        hhtm = '<div class="h_msg">'+self.name+' has been translated into'+lang+'sir!</div>'
        
        self.present(hhtm)
        
        
    def createTSVs(self, folder="data"):
        self.fileCount = 0
        for index, r in self.data.iterrows():
            data_temp = self.data.loc[index:index]
            self.fileCount +=1
            data_temp.to_csv(self.dataFolder+'/doc_'+str(index+1)+'.tsv', sep='\t', index=False, header=False)
        hhtm = '<div class="h_msg">TSV files for '+self.name+' have been created under '+self.dataFolder+' folder, sir!</div>'
        
        self.present(hhtm)
        

    # get all unique words in all documents and store them in a list words[]
    def getUniqueTerms(self):
        for i in range(1,self.fileCount):
            p = Property(i)
            terms = p.getUniqueTerms()
            for word in terms:
                if word not in self.words:
                    self.words.append(word)
        return(self.words)
    
    # build the vocabulary
    def buildVocabulary(self, fileName='vocabulary.csv'):
        self.vocabularyFile = str(fileName)
        self.words = self.getUniqueTerms()
        wordID=0
        with open(self.vocabularyFile,'wb') as vfile:
            for i in range(0,len(self.words)):
                vfile.write(str(wordID).encode())
                vfile.write(str('\t').encode())
                vfile.write(str(self.words[i]).encode())
                vfile.write('\n'.encode())
                wordID+=1
        vfile.close()
        
        hhtm = '<div class="h_msg">Vocabulary file for '+self.name+' has been created in '+self.vocFile+', sir!</div>' 
        self.present(hhtm)
    
    # creating index-file by loading vocabulary file and then comparing files with all vocabularys
    def getVocabulary(self):
        with open(self.vocabularyFile, newline='') as file:
            reader = csv.reader(file, delimiter='\t')
            self.vocabulary = np.array(list(map(tuple, reader)))
        file.close()
        return(self.vocabulary)
    
    # function to get ID of a given term (this one uses the voc array instead of reading from file)
    def getID(self,term):
        for row in self.vocabulary:
            if row[1] == term:
                return(row[0])
        return(-1)
    # function to get Term for a given term ID
    def getTerm(self,tid):
        for row in self.vocabulary:
            if row[0] == tid:
                return(row[1])
        return('')




# class for Property
class Property:
    def __init__(self, fileId, fromFile = False, path = 'data'):
        self.fileId = fileId
        self.terms = []
        self.uniqueTerms = []
        self.path = path
        self.title = ''
        self.description = ''
        self.rooms = 0
        self.url = 0
        self.lat = 0
        self.long = 0
        self.city = 0
        if fromFile == True:
            self.getFromTSV()
        
    def getFromTSV(self):    
        data = [i.strip('\n').split('\t') for i in open(self.path+'/doc_'+str(self.fileId)+'.tsv',encoding="utf8")]
        self.title = data[0][7].replace('\\n', ' ')
        self.description = data[0][4].replace('\\n', ' ')
        for w in (cleanData(self.title)):
            self.terms.append(w)
            if w not in self.uniqueTerms: self.uniqueTerms.append(w)
        for w in (cleanData(self.description)):
            self.terms.append(w)
            if w not in self.uniqueTerms: self.uniqueTerms.append(w) 
        
        self.price = float(str(data[0][0]).replace('$','')) 
        self.rooms = float(data[0][1].replace('Studio','1'))
        self.url = data[0][8]
        self.lat = data[0][5]
        self.long = data[0][6]
        self.city = data[0][2]
        
        
        
        
        
    # function to get unique words from a TSV file given its ID
    def getUniqueTerms(self):
            data = [i.strip('\n').split('\t') for i in open(self.path+'/doc_'+str(self.fileId)+'.tsv',encoding="utf8")]
            for w in (cleanData(data[0][4].replace('\\n', ' '))):
                if w not in self.uniqueTerms: self.uniqueTerms.append(w)
            for w in (cleanData(data[0][7].replace('\\n', ' '))):
                if w not in self.uniqueTerms: self.uniqueTerms.append(w)
            
            return(list(set(self.uniqueTerms)))
     
        
    # funciton retuns the details of a file given file id

    def getDetailsSimple(self):
        return([self.fileId, self.title, self.description, self.city, self.url])
    
    
class HoohleSimple:
    def __init__(self, airbnb, buildIndex = False):
        self.data = airbnb
        self.index = {}
        if buildIndex:
            for f in range(1,self.data.fileCount+1):
                p= Property(f,True)
                for t in p.uniqueTerms:
                    termID = self.data.getID(t)
                    if termID in self.index:
                        self.index[termID].append(f)
                    else:
                        self.index[termID]=[f]
    
    # two functions to save and read index to/from a file
    def saveIndex(self, file):
        deepdish.io.save(file, self.index)
    def readIndex(self, file):
        self.index = deepdish.io.load(file)
        
    # get list of docs contains a given term ID
    def getDocsByID(self, termId):
        return (self.index[str(termId)])
    # get list of docs contains a given term 
    def getDocsByTerm(self, term):
        return (self.index[str(self.data.getID(term))])
    
    # Search Engine 1
    def getDocsByQuery(self, q):
        ts = cleanData(q)
        docsIDs = self.getDocsByTerm(ts[0])
        
        if docsIDs == None : return []
        if len(ts)>1:
            for t in ts[1:]:
                tempDocs = self.getDocsByTerm(t)
                if tempDocs != None:
                        docsIDs = set(docsIDs).intersection(tempDocs)
                else: return None
        return(list(docsIDs))
    
#    def readIndex(self, index):
#        self.index = index

    def printResult(self, property):
        rhtm ='<div class="h_result"><div class="h_title"><a href="'+property[3]+'">'+property[1]+'</a></div>'
        rhtm +='<div class="h_link"><a href="'+property[3]+'">'+property[3]+'</a></div>'
        rhtm +='<div class="h_city">'+str(r[2])+'</div>'
        rhtm +='<div class="h_disc">'+property[2].replace('\n', '')[:300]+'... </div>'
        rhtm +='</div>'
        return (rhtm)
    
    def printResults(self, query):
        docs = self.getDocsByQuery(query)
        styles = open("style/style.css", "r").read()
        hhtm = '<style>%s</style>' % styles     
        hhtm +='<div class="Hoohle" ><div class="h_results">'
        if docs != None:
            for doc in docs:
                p = Property(doc, True)
                hhtm += self.printResult(p.getDetailsSimple())
        else:
            hhtm +='<div class="h_sorry">Sorry! No results found for your request!</div>'
        hhtm += '</div></div>'
        display(HTML(hhtm))

        
class HoohleTFIDF(HoohleSimple):
    def __init__(self, airbnb, buildIndex=False):
        HoohleSimple.__init__(self, airbnb)
        
        self.wordsAll = []
        
        # Create the new inverted index
        #print("building the inverted index, this could take few minutes.. please wait!")
        self.indexTFIDF = {}
        if buildIndex:
            for termID in list(self.index.keys()):
                self.indexTFIDF[termID]=[]
                for docID in self.index[termID]:
                    self.indexTFIDF[termID].append(self.get_TFIDF(docID, termID))

        
    # two functions to save and read index to/from a file
    def saveIndex(self, file):
        deepdish.io.save(file, self.indexTFIDF)
    def readIndex(self, file):
        self.indexTFIDF = deepdish.io.load(file)   
    
    # for a given (file id, term), returns the term frequency in this file
    def get_TF(self,fid, term):
        p = Property(fid, True)
        words = p.terms
        return (words.count(term)/len(words))

    # returns the IDF for a given term in the whole data we have
    def get_IDF(self, term):
        return (math.log(self.data.fileCount/len(self.index[self.data.getID(term)])))
    
    # returns the TFIDF for a given (file id, term)
    def get_TFIDF(self, fid, termID):
        #getting TF for term in document with document id = fid and multiply it with the IDF of the term
        term = self.data.vocabulary[int(termID)][1]
        return ([fid, self.get_TF(fid, term) * self.get_IDF(term)])
    

     # function to get the cosine similarity between two given vectors

    def getCosineSimi(self, v1,v2):
         return (1 - spatial.distance.cosine(v1, v2))
    
    # function to get the tfidf for a given query
    def getTFIDF_query(self, q):
        terms = cleanData(q)
        return([(terms.count(w))*math.log(self.data.fileCount+1/(1+len(self.index[self.data.getID(w)]))) for w in list(set(terms))])
    
    # function to get the tfidf vector for a given query in a document 
    def getTFIDF_vector(self, doc, terms):
        v=[]
        p = Property(doc, True)
        
        for term in list(set(cleanData(terms))):
            if term not in p.uniqueTerms: v.append(0)
            else:
                l = self.indexTFIDF[str(self.data.getID(term))]
                for i in l:
                    if i[0]==doc:
                        v.append(i[1])
        return(v)
    
    # function to get items with repitition more or equal to a threshold from a given list
    def getNdocs(self, l, n):
        docs = []
        docs = list(filter(lambda x:l.count(x)>=n, l))
        return(docs)
    
    # Search Engine 2
    def getDocsByQuery(self, q, k=10):
        ts = cleanData(q)
        #docsIDs = self.getDocsByID(self.data.getID(ts[0]))
        docsIDs = self.index[self.data.getID(ts[0])]
        
        if docsIDs == None : return None
        if len(ts)>1:
            for t in ts[1:]:
                tid = self.data.getID(t)
                if tid != -1:
                    tempDocs = self.index.get(tid)
                    if tempDocs != None:
                        docsIDs = set(docsIDs).intersection(tempDocs)
                else: return None
        if len(docsIDs)<k:
            kDocs = self.completeToK(q,docsIDs,k)
            docsIDs = list(docsIDs)
            for kd in kDocs:
                docsIDs.append(kd[0])

        # sorting the results docs by their similarity to the query
        smlrtyDocs =[]
        for doc in docsIDs:
            smlrtyDocs.append([doc, self.getCosineSimi(self.getTFIDF_query(q), self.getTFIDF_vector(doc,q))])

        return(hq.nlargest(k, smlrtyDocs, key=lambda x:x[1]))
    
    # function (in case we got results less than K) this function will complete the resulted documents to k from the heap
    def completeToK(self, qq, r,k):
        if len(r)>k: return None
        rk = []
        qTFIDF = self.getTFIDF_query(qq)
        l = []
        for term in cleanData(qq):
            l.extend(self.index[self.data.getID(term)])

        if len(l)>k:
            l = self.getNdocs(l, len(cleanData(qq))-1)

        for d in list(set(l)):

            if d not in r:
                cosS = self.getCosineSimi(qTFIDF, self.getTFIDF_vector(d,qq))
                if cosS != 0:
                    rk.append([d, cosS])

        return(hq.nlargest(k-len(r), rk, key=lambda x:x[1]))

    # funciton retuns the details of a file given file id

    def getDetails(self, fid, q):
        detlist =[]
        smlrty = self.getCosineSimi(self.getTFIDF_query(q), self.getTFIDF_vector(fid,q))
        p = Property(fid, True)
        
        return([p.title, p.description, p.city, p.url, smlrty])
    
    def printResult(self, r):
        rhtm ='<div class="h_result"><div class="h_title"><a href="'+str(r[3])+'">'+str(r[0])+'</a></div>'
        rhtm +='<div class="h_link"><a href="'+str(r[3])+'">'+str(r[3])+'</a></div>'
        rhtm +='<div class="h_city">City: '+str(r[2])+'</div>'
        rhtm +='<div class="h_disc">'+r[1].replace('\n', '')[:300]+'... </div>'
        rhtm +='<div class="h_score"> Similarity Score: '+str(r[4])+'... </div>'
        rhtm +='</div>'
        return (rhtm)
    
    def printResults(self, query, k=10):
        docs = self.getDocsByQuery(query, k)
        styles = open("style/style.css", "r").read()
        hhtm = '<style>%s</style>' % styles     
        hhtm +='<div class="Hoohle" ><div class="h_results">'
        if docs != None:
            for doc in docs:
                hhtm += self.printResult(self.getDetails(doc[0], query))
        else:
            hhtm +='Sorry! No results found for your request!'
        hhtm += '</div></div>'
        display(HTML(hhtm))
        

# class HoohleNOSTRO
class HoohleNOSTRO(HoohleSimple):
    def __init__(self, airbnb, buildIndex=False):
        HoohleSimple.__init__(self, airbnb)
        self.data = airbnb
        self.data.data['price'] = [float(str(i).replace('$','')) for i in self.data.data['average_rate_per_night']]
        self.maxPrice = self.data.data['price'].max()
        self.minPrice = self.data.data['price'].min()
        
        self.data.data['rooms'] = self.data.data['bedrooms_count'].replace({'Studio':'1'}).astype(float).fillna(0.0)
        self.maxRooms = self.data.data['rooms'].max()
        self.minRooms = self.data.data['rooms'].min()
        
        if buildIndex: self.buildIndex()

        

             
    # two functions to save and read index to/from a file
    def saveIndex(self, file):
        deepdish.io.save(file, self.indexNostro)
    def readIndex(self, file):
        self.indexNostro = deepdish.io.load(file)  
    def saveGeoDisct(self):
        self.geodict = deepdish.io.load(file)
        
    # Price scoring algorithm: is taking the max price as the value with score 1, and the min value with score = 0
    # NOTE: it doesnt reflect any real absolute value, 
    # but when it's combined with the price score of the query it allow us to get the similarity between the result and the query
    def scoringPrice(self, pr):
        if pr == 0 or pr == None: return (0)
        if pr >= self.maxPrice: return (1)
        return((pr - self.minPrice)/(self.maxPrice - self.minPrice))
    
    # same explaination
    def scoringRoom(self, r):
        if r == 0 or r == None: return (0)
        if r >= self.maxRooms : return (1)
        return ((r - self.minRooms)/(self.maxRooms - self.minRooms))
    
    def buildGeoInfo(self):
        from geopy.geocoders import Nominatim
        self.geolocator = Nominatim(user_agent="ADM201-HW3")
        from geopy.exc import GeocoderTimedOut
    
        # getting the list of unique cities
        self.uniqueCity = self.data.data['city'].unique()
        
        self.geodict = {}
        for city in self.uniqueCity:
            try:
                location = self.geolocator.geocode(city+" TX", timeout=10)
                if location != None:
                    self.geodict[city] = [location.latitude, location.longitude]
                else:
                    self.geodict[city] = [None, None]
            except GeocoderTimedOut as e:
                print("Error: geocode failed ")
        
        #some values couldnt be retrieved by function, so we got them manually: 
        self.geodict['North Padre Island Corpus Christi'] = [27.800583,-97.396378]
        self.geodict['诺斯莱克'] = [33.0825,-97.253056]
        self.geodict['Boliver peninsula'] = [29.562353,-94.394371]
        self.geodict['Bolivar Pennisula'] = [29.562353,-94.394371]
        self.geodict['阿纳瓦克'] = [29.767830262 ,-94.67416397]
    
    def insetGeoInfo(self):
        self.data.data['ccLat']= [self.geodict[x][0] for x in self.data.data.city]
        self.data.data['ccLong']= [self.geodict[x][1] for x in self.data.data.city]
        
        self.data.data['distanceToCC'] = self.data.data.apply(lambda row: geopy.distance.vincenty((row.latitude,row.longitude), (row.ccLat,row.ccLong)).km, axis=1)
        
        
        # creating the new index
        
    def buildIndex(self):
        self.indexNostro = {}
        for index,row in self.data.data.iterrows():
            self.indexNostro[index+1] = [self.scoringPrice(row.price), self.scoringRoom(int(row.rooms)), self.scoringLoc(row.city, row.distanceToCC)]
    
    # location scores
    def scoringLoc(self, city, dista):
        maxDist = self.data.data[self.data.data['city']==city]['distanceToCC'].max()
        if dista >= maxDist: return (0.0)
        return (1 - (dista/maxDist))
    
    # function to order results by their ranking
    def rank(self, vv, res):
        resHeap = []
        for doc in res:
            resHeap.append([doc, getCosineSimi(self.indexNostro[doc], vv)])
        return(hq.nlargest(len(resHeap), resHeap, key=lambda x:x[1]))
    
    
    # function to calculate the score of user input
    def gimmeTheScore(self, vv):
        return ([self.scoringPrice(vv[0]), self.scoringRoom(vv[1]), self.scoringLoc(vv[2], vv[3])])
    
    
    # funciton retuns the details of a file given file id

    def getDetails(self, fid):
        detlist =[]
        p = Property(fid, True)
        
        return([p.title, p.description, p.city, p.url])
    
    def showUI(self):
        qq = input("What are you looking for?")
        qprice = input("price?")
        qrooms = input("number of rooms?")
        qcity = input("In which city?")
        qdist = input("distance to city center?")
        self.printResults(self, qq, [float(qprice), float(qrooms), qcity, float(qdist)])
        
    def printResult(self, r,rank):
        rhtm ='<div class="h_result"><div class="h_rank">'+str(rank)+'</div>'
        rhtm +='<div class="h_title"><a href="'+r[3]+'">'+r[0]+'</a></div>'
        rhtm +='<div class="h_link"><a href="'+r[3]+'">'+r[3]+'</a></div>'
        rhtm +='<div class="h_city">City: '+str(r[2])+'</div>'
        rhtm +='<div class="h_disc">'+r[1].replace('\n', '')[:300]+'... </div>'
        rhtm +='</div>'
        return (rhtm)

    def printResults(self, query, v):
        res = self.getDocsByQuery(query)
        vv = self.gimmeTheScore(v)
        docs = self.rank(vv, res)
        styles = open("style/style.css", "r").read()
        hhtm = '<style>%s</style>' % styles     
        hhtm +='<div class="Hoohle" ><div class="h_results">'
        if docs != None:
            rank=1
            for doc in docs:
#                p = Property(doc[0], True)
                hhtm +=self.printResult(self.getDetails(doc[0]),rank)
                rank+=1
        else:
            hhtm +='Sorry! No results found for your request!'
        hhtm += '</div></div>'
        display(HTML(hhtm))
        
