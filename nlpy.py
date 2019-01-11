
# pip install matplotlib
# pip intsall WordCloud - wordcloud
# rqeuires Microsoft Visual C++ 14.0 is required.  Visual c++ Build tools

# error: Microsoft Visual C++ 14.0 is required. Get it with "Microsoft Visual C++ Build Tools": http://landinghub.visualstudio.com/visual-cpp-build-tools

# https://visualstudio.microsoft.com/downloads/#build-tools-for-visual-studio-2017
# https://www.microsoft.com/en-in/download/details.aspx?id=48159

#https://www.scivision.co/python-windows-visual-c++-14-required/ 

#https://www.lfd.uci.edu/~gohlke/pythonlibs/

#pip install C:/some-dir/some-file.whl 

#C:\Users\pavan>pip install c:\users\pavan\downloads\wordcloud-1.5.0-cp37-cp37m-win32.whl

import nltk
import matplotlib.pyplot as plt
from wordcloud import WordCloud

text = "all your base are belong to us"

wordcloud = WordCloud().generate(text)

def plot_wordcloud(wordcloud):
    plt.imshow(wordcloud)
    plt.axis("off")
    plt.show()

#wordcloud = WordCloud(stopwords={'to', 'of'}).generate(text)
#plot_wordcloud(wordcloud)
import numpy as np
from PIL import Image
from os import path

img = Image.open(path.join(path.dirname(__file__), "supplementary_files",   "c:\\users\\pavan\\downloads\\washington2.jpg"))
mask = np.array(img)

text_base = "That swag continues throughout the film, a revenge saga filled with long action blocks and some solid surprises. Kaali (Rajinikanth) is recommended for a college warden’s post and he arrives to set things that aren’t in order in the institution. But tackling ragging and college politics isn’t just what he’ll have to face — he has a past, one filled with secrets, and one that will return to haunt him."

from wordcloud import STOPWORDS

stopwords = set(STOPWORDS)
stopwords.add("that")
stopwords.add("the")
wordcloud = WordCloud(relative_scaling=1.0, stopwords=stopwords, max_words=1000, mask=mask).generate(text_base)
#plot_wordcloud(wordcloud)

##############################

from nltk.corpus import stopwords
stopWords = set(stopwords.words('english'))

from nltk.text import Text
aText = Text(nltk.corpus.gutenberg.words('c:\\users\\pavan\\downloads\\sampleText.txt'))
fdist = nltk.FreqDist(aText)
stopwords = nltk.corpus.stopwords.words('english')
newStopWords = ['has', '896', '2018', '19','26', '108', '43', '922',".","?"]
stopwords.extend(newStopWords)

# freqdistPlot
#fdist.plot(50, cumulative=True)
fdist_no_punc_no_stopwords = nltk.FreqDist(dict((word, freq) for word, freq in fdist.items() if word not in stopwords and word.isalpha()))
fdist_no_punc_no_stopwords.plot(50, cumulative=False, title="50 most common tokens (no stopwords or punctuation)")



#print(fdist)
#print(fdist.hapaxes())
# Collocations
# A collocation is a pair or group of words that are habitually juxtaposed. For instance “red wine”
#print(aText.collocations())
#print(aText)



#######################################

from nltk.corpus import wordnet as wn

print(wn.synset('house_cat.n.01').hypernyms())
car = wn.synset('car.n.01')
boat = wn.synset('boat.n.01')
print(car.path_similarity(boat))

#########################
from nltk.stem import PorterStemmer
porter = PorterStemmer()
word_list = ["connected", "connecting", "connection", "connections"]

from nltk.tokenize import sent_tokenize, word_tokenize
Xword_list = word_tokenize(text_base)



for word in Xword_list:
    #print(porter.stem(word))
    print(word)

print("##############")
from nltk.stem import WordNetLemmatizer
lemmatizer = WordNetLemmatizer()


for word in Xword_list:
    print(lemmatizer.lemmatize(word))
    



































