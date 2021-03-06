# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
import nltk
from nltk.tokenize import word_tokenize
from nltk import pos_tag
from nltk.corpus import stopwords
from sklearn.preprocessing import LabelEncoder
from collections import defaultdict
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn import model_selection, naive_bayes, svm
from sklearn.metrics import accuracy_score
import unicodedata
import re
import pickle

GREEK_STOP = ['αδιακοπα','αι','ακομα','ακομη','ακριβως','αληθεια','αληθινα','αλλα','αλλαχου','αλλες','αλλη','αλλην','αλλης','αλλιως','αλλιωτικα','αλλο','αλλοι','αλλοιως','αλλοιωτικα','αλλον','αλλος','αλλοτε','αλλου','αλλους','αλλων','αμα','αμεσα','αμεσως','αν','ανα','αναμεσα','αναμεταξυ','ανευ','αντι','αντιπερα','αντις','ανω','ανωτερω','αξαφνα','απ’','απεναντι','απο','αποψε','αρα','αραγε','αργα','αργοτερο','αρκετα','αρχικα','ας','αυριο','αυτα','αυτες','αυτη','αυτην','αυτης','αυτο','αυτοι','αυτον','αυτος','αυτου','αυτους','αυτων','αφοτου','αφου','βεβαια','βεβαιως','βεβαιοτατα','γι’','για','γρηγορα','γυρω','∆α','∆ε','∆εινα','∆εν','∆εξια','∆ηθεν','∆ηλα∆η','δεν', 'δες', 'δεσ','δι’','δια','διαρκως','δικα','δικο','δικοι','δικοι','δικου','δικους','διολου','διπλα','διχως','εαν','εαυτο','εαυτον','εαυτου','εαυτους','εαυτων','εγκαιρα','εγκαιρως','εγω','εδω','ειδεμη','ειθε','ειμαι','ειμαστε','ειναι','εις','εισαι','εισαστε','ειστε','ειτε','ειχα','ειχαμε','ειχαν','ειχατε','ειχε','ειχες','εκαστα','εκαστες','εκαστη','εκαστην','εκαστης','εκαστο','εκαστοι','εκαστον','εκαστος','εκαστου','εκαστους','εκαστων','εκει','εκεινα','εκεινες','εκεινη','εκεινην','εκεινης','εκεινο','εκεινοι','εκεινον','εκεινος','εκεινου','εκεινους','εκεινων','εκτος','εμας','εμεις','εμενα','εμπρος','εν','ενα','εναν','ενας','ενος','εντελως','εντος','εντωμεταξυ','ενω','εξ','εξαφνα','εξης','εξισου','εξω', 'εο', 'επανω','επειδη','επειτα','επι','επισης','επομενως','εσας','εσεις','εσενα','εστω','εσυ','ετερα','ετεραι','ετερας','ετερες','ετερη','ετερης','ετερο','ετεροι','ετερον','ετερος','ετερου','ετερους','ετερων','ετουτα','ετουτες','ετουτη','ετουτην','ετουτης','ετουτο','ετουτοι','ετουτον','ετουτος','ετουτου','ετουτους','ετουτων','ετσι','ευγε','ευθυς','ευτυχως','εφεξης','εχει','εχεις','εχετε','εχθες','εχομε','εχουμε','εχουν','εχτες','εχω','εως','Η','η','η','ηδη','ημασταν','ημαστε','ημουν','ησασταν','ησαστε','ησουν','ηταν','ητανε','ητοι','ηττον','θα','ι','ιδια','ιδιαν','ιδιας','ιδιες','ιδιο','ιδιοι','ιδιον','ιδιος','ιδιου','ιδιους','ιδιων','ιδιως','ισαμε','ισια','ισως','καθε','καθεμια','καθεμιας','καθενα','καθενας','καθενος','καθετι','καθολου','καθως','και','κακα','κακως','καλα','καλως','καμια','καμιαν','καμιας','καμποσα','καμποσες','καμποση','καμποσην','καμποσης','καμποσο','καμποσοι','καμποσον','καμποσος','καμποσου','καμποσους','καμποσων','κανεις', 'κανει', 'καντε', 'καναμε','κανενα','κανεναν','κανενας','κανενος','καποια','καποιαν','καποιας','καποιες','καποιο','καποιοι','καποιον','καποιος','καποιου','καποιους','καποιων','καποτε','καπου','καπως','κατ’','κατα','κατι','κατιτι','κατοπιν','κατω','κιολας','κλπ.','κοντα','κτλ.','κυριως','λιγακι','λιγο','λιγοτερο','λογω','λοιπα','λοιπον','μα','μαζι','μακαρι','μακρια','μαλιστα','μαλλον','μας','με','μεθαυριο','μειον','μελει','μελλεται','μεμιας','μεν','μερικα','μερικες','μερικοι','μερικους','μερικων','μεσα','μετ','μετα','μεταξυ','μεχρι','μη','μηδε','μην','μηπως','μητε','μια','μιαν','μιας','μολις','μολονοτι','μοναχα','μονες','μονη','μονη','μονης','μονο','μονοι','μονομιας','μονος','μονου','μονους','μονων','μου','μπορει','μπορουν','μπραβο','μπρος','να','ναι','νωρις','ξανα','ξαφνικα','ο', 'οε', 'οι','ολα','ολες','ολη','ολην','ολης','ολο','ολογυρα','ολοι','ολον','ολος','ολοτελα','ολου','ολους','ολων','ολως','ολωσδιολου','ομως','οποια','οποιαδηποτε','οποιαν','οποιανδηποτε','οποιας','οποιοδηποτε','οποιες','οποιεσδηποτε','οποιο','οποιοδηποτε','οποιοι','οποιον','οποιονδηποτε','οποιος','οποιοσδηποτε','οποιου','οποιουδηποτε','οποιους','οποιων','οποιωνδηποτε','οποτε','οποτεδηποτε','οπου','οπουδηποτε','οπως','ορισμενα','ορισμενες','ορισμενων','ορισμενως','οσα','οσαδηποτε','οσες','οσεσδηποτε','οση','οσηδηποτε','οσην','οσηνδηποτε','οσης','οσησδηποτε','οσο','οσοδηποτε','οσοι','οσοιδηποτε','οσον','οσονδηποτε','οσος','οσοσδηποτε','οσου','οσουδηποτε','οσους','οσουσδηποτε','οσων','οσωνδηποτε','οταν','οτι','οτιδηποτε','οτου','ου','ουδε','ουτε','παλι','παντοτε','παντου','παντως','παρα','περα','περι','περιπου','περισσοτερο','περσι','περυσι','πια','πιθανον','πιο','πισω','πλαι','πλεον','πλην','ποια','ποιαν','ποιας','ποιες','ποιο','ποιοι','ποιον','ποιος','ποιου','ποιους','ποιων','πολυ','ποσες','ποση','ποσην','ποσης','ποσοι','ποσος','ποσους','ποτε','που','πουθε','πουθενα','πρεπει','πριν','προ','προκειμενου','προκειται','προπερσι','προς','προτου','προχθες','προχτες','πρωτυτερα','πως','σαν','σας','σε','σεις','σημερα','σιγα','σου','στα','στη','στην','στης','στις','στο','στον','στου','στους','στων','συγχρονως','συν','συναμα','συνεπως','συνηθως','συχνα','συχνες','συχνη','συχνην','συχνης','συχνο','συχνοι','συχνον','συχνος','συχνου','συχνους','συχνων','σχεδον','σωστα','τα','ταδε','ταυτα','ταυτες','ταυτη','ταυτην','ταυτης','ταυτο','ταυτον','ταυτος','ταυτου','ταυτων','ταχα','ταχατε','τελικα','τελικως','τες','τετοια','τετοιαν','τετοιας','τετοιες','τετοιο','τετοιοι','τετοιον','τετοιος','τετοιου','τετοιους','τετοιων','τη','την','της','τι','τιποτα','τιποτε','τις','το','τοι','τον','τος','τοσα','τοσες','τοση','τοσην','τοσης','τοσο','τοσοι','τοσον','τοσος','τοσου','τοσους','τοσων','τοτε','του','τουλαχιστο','τουλαχιστον','τους','τουτα','τουτες','τουτη','τουτην','τουτης','τουτο','τουτοι','τουτοις','τουτον','τουτος','τουτου','τουτους','τουτων','τυχον','των','τωρα','υπ’','υπερ','υπο','υποψη','υποψιν','υστερα','φετος','χαμηλα','χθες','χτες','χωρις','χωριστα','ψηλα','ω','ωραια','ως','ωσαν','ωσοτου','ωσπου','ωστε','ωστοσο','ωχ', '-', '+', '=', '«', '»', '(', ')', '[', ']', ';', '?', '<', '> ', 'βιντεο', 'video', 'vid', 'RT', 'retweet' ]
tags = ['!', '"', '$', '%', '&', '\'', '(', ')', '*', '+', ',', '-', '.', '/', ':', ';', '<', '=', '>', '?', '[', ']', '^', '_', '`', '{', '|', '}', '~', '«', '»', '€']


df = pd.read_csv(r"basket_dataset.csv",encoding='utf-8')
#df=df[:50]

#Data Pre-processing - This will help in getting better results through the classification algorithms
# remove blank rows
df['content'].dropna(inplace=True)

# text to lower case
df['content'] = [entry.lower() for entry in df['content']]

df2 = pd.DataFrame(columns=['content_2'])
i = []
#bgale ta shmeia sti3is
for entry in df['content']:
    entry = entry.encode("utf-8")
    for tag in tags:
        entry = entry.replace(tag, '')
    i.append(entry.decode('utf-8'))
df['content']=i

# tokenization
df['content']= [', '.join(word_tokenize(entry)) for entry in df['content']]

#Afairesh tonwn
def strip_accents(s):
   return ''.join(c for c in unicodedata.normalize('NFD', s)
                  if unicodedata.category(c) != 'Mn')

#a=strip_accents(entry)
df['content']=[(strip_accents(entry)) for entry in df['content']]

df['content']= [entry.split(",") for entry in df['content']]

#remove stop words
for index,entry in enumerate(df['content']):
    final_words = []
    for word in entry:
        if word.encode('utf-8') not in GREEK_STOP:
            final_words.append(word)
    # The final processed set of words
    df.loc[index,'text_final'] = str(final_words)

# split the model into Train and Test Data set
Train_X, Test_X, Train_Y, Test_Y = model_selection.train_test_split(df['text_final'],df['basket'],test_size=0.3)

# label encode the target variable - transform categorical data of string type in the data set into numerical values
Encoder = LabelEncoder()
Train_Y = Encoder.fit_transform(Train_Y)
Test_Y = Encoder.fit_transform(Test_Y)

# vectorize the words by using TF-IDF Vectorizer - to find how important a word in document is in comaprison to df
Tfidf_vect = TfidfVectorizer(max_features=5000)
Tfidf_vect.fit(df['text_final'])

############### apo8ikeush modelou
pickle.dump(Tfidf_vect, open("vectorizer.pickle", "wb"))

Train_X_Tfidf = Tfidf_vect.transform(Train_X)
Test_X_Tfidf = Tfidf_vect.transform(Test_X)

# run different algorithms to classify out data check for accuracy

# Classifier - Algorithm - Naive Bayes
# fit the training dataset on the classifier
Naive = naive_bayes.MultinomialNB()
Naive.fit(Train_X_Tfidf,Train_Y)

NB_clf = 'NB_model.sav'
pickle.dump(Naive, open(NB_clf, 'wb'))

# predict the labels on validation dataset
predictions_NB = Naive.predict(Test_X_Tfidf)
predictions_NB_proba = Naive.predict_proba(Test_X_Tfidf)

# Use accuracy_score function to get the accuracy
print("Naive Bayes Accuracy Score -> ",accuracy_score(predictions_NB, Test_Y)*100)

# Classifier - Algorithm - SVM
# fit the training dataset on the classifier
SVM = svm.SVC(C=1.0, kernel='linear', degree=3, gamma='auto', probability=True)
SVM.fit(Train_X_Tfidf,Train_Y)

SVM_clf = 'SVM_model.sav'
pickle.dump(SVM, open(SVM_clf, 'wb'))

# predict the labels on validation dataset
predictions_SVM = SVM.predict(Test_X_Tfidf)
predictions_SVM_proba = SVM.predict_proba(Test_X_Tfidf)

# Use accuracy_score function to get the accuracy
print("SVM Accuracy Score -> ",accuracy_score(predictions_SVM, Test_Y)*100)
