# -*- coding: utf-8 -*-

import requests
from bs4 import BeautifulSoup
import time
import os
import numpy as np
from nltk.tokenize import word_tokenize
import unicodedata
import re
import pickle

GREEK_STOP = ['αδιακοπα', 'αι', 'ακομα', 'ακομη', 'ακριβως', 'αληθεια', 'αληθινα', 'αλλα', 'αλλαχου', 'αλλες',
			  'αλλη', 'αλλην', 'αλλης', 'αλλιως', 'αλλιωτικα', 'αλλο', 'αλλοι', 'αλλοιως', 'αλλοιωτικα', 'αλλον',
			  'αλλος', 'αλλοτε', 'αλλου', 'αλλους', 'αλλων', 'αμα', 'αμεσα', 'αμεσως', 'αν', 'ανα', 'αναμεσα',
			  'αναμεταξυ', 'ανευ', 'αντι', 'αντιπερα', 'αντις', 'ανω', 'ανωτερω', 'αξαφνα', 'απ’', 'απεναντι',
			  'απο', 'αποψε', 'αρα', 'αραγε', 'αργα', 'αργοτερο', 'αρκετα', 'αρχικα', 'ας', 'αυριο', 'αυτα',
			  'αυτες', 'αυτη', 'αυτην', 'αυτης', 'αυτο', 'αυτοι', 'αυτον', 'αυτος', 'αυτου', 'αυτους', 'αυτων',
			  'αφοτου', 'αφου', 'βεβαια', 'βεβαιως', 'βεβαιοτατα', 'γι’', 'για', 'γρηγορα', 'γυρω', '∆α', '∆ε',
			  '∆εινα', '∆εν', '∆εξια', '∆ηθεν', '∆ηλα∆η', 'δεν', 'δες', 'δεσ', 'δι’', 'δια', 'διαρκως', 'δικα',
			  'δικο', 'δικοι', 'δικοι', 'δικου', 'δικους', 'διολου', 'διπλα', 'διχως', 'εαν', 'εαυτο', 'εαυτον',
			  'εαυτου', 'εαυτους', 'εαυτων', 'εγκαιρα', 'εγκαιρως', 'εγω', 'εδω', 'ειδεμη', 'ειθε', 'ειμαι',
			  'ειμαστε', 'ειναι', 'εις', 'εισαι', 'εισαστε', 'ειστε', 'ειτε', 'ειχα', 'ειχαμε', 'ειχαν', 'ειχατε',
			  'ειχε', 'ειχες', 'εκαστα', 'εκαστες', 'εκαστη', 'εκαστην', 'εκαστης', 'εκαστο', 'εκαστοι', 'εκαστον',
			  'εκαστος', 'εκαστου', 'εκαστους', 'εκαστων', 'εκει', 'εκεινα', 'εκεινες', 'εκεινη', 'εκεινην',
			  'εκεινης', 'εκεινο', 'εκεινοι', 'εκεινον', 'εκεινος', 'εκεινου', 'εκεινους', 'εκεινων', 'εκτος',
			  'εμας', 'εμεις', 'εμενα', 'εμπρος', 'εν', 'ενα', 'εναν', 'ενας', 'ενος', 'εντελως', 'εντος',
			  'εντωμεταξυ', 'ενω', 'εξ', 'εξαφνα', 'εξης', 'εξισου', 'εξω', 'εο', 'επανω', 'επειδη', 'επειτα',
			  'επι', 'επισης', 'επομενως', 'εσας', 'εσεις', 'εσενα', 'εστω', 'εσυ', 'ετερα', 'ετεραι', 'ετερας',
			  'ετερες', 'ετερη', 'ετερης', 'ετερο', 'ετεροι', 'ετερον', 'ετερος', 'ετερου', 'ετερους', 'ετερων',
			  'ετουτα', 'ετουτες', 'ετουτη', 'ετουτην', 'ετουτης', 'ετουτο', 'ετουτοι', 'ετουτον', 'ετουτος',
			  'ετουτου', 'ετουτους', 'ετουτων', 'ετσι', 'ευγε', 'ευθυς', 'ευτυχως', 'εφεξης', 'εχει', 'εχεις',
			  'εχετε', 'εχθες', 'εχομε', 'εχουμε', 'εχουν', 'εχτες', 'εχω', 'εως', 'Η', 'η', 'η', 'ηδη', 'ημασταν',
			  'ημαστε', 'ημουν', 'ησασταν', 'ησαστε', 'ησουν', 'ηταν', 'ητανε', 'ητοι', 'ηττον', 'θα', 'ι', 'ιδια',
			  'ιδιαν', 'ιδιας', 'ιδιες', 'ιδιο', 'ιδιοι', 'ιδιον', 'ιδιος', 'ιδιου', 'ιδιους', 'ιδιων', 'ιδιως',
			  'ισαμε', 'ισια', 'ισως', 'καθε', 'καθεμια', 'καθεμιας', 'καθενα', 'καθενας', 'καθενος', 'καθετι',
			  'καθολου', 'καθως', 'και', 'κακα', 'κακως', 'καλα', 'καλως', 'καμια', 'καμιαν', 'καμιας', 'καμποσα',
			  'καμποσες', 'καμποση', 'καμποσην', 'καμποσης', 'καμποσο', 'καμποσοι', 'καμποσον', 'καμποσος',
			  'καμποσου', 'καμποσους', 'καμποσων', 'κανεις', 'κανει', 'καντε', 'καναμε', 'κανενα', 'κανεναν',
			  'κανενας', 'κανενος', 'καποια', 'καποιαν', 'καποιας', 'καποιες', 'καποιο', 'καποιοι', 'καποιον',
			  'καποιος', 'καποιου', 'καποιους', 'καποιων', 'καποτε', 'καπου', 'καπως', 'κατ’', 'κατα', 'κατι',
			  'κατιτι', 'κατοπιν', 'κατω', 'κιολας', 'κλπ.', 'κοντα', 'κτλ.', 'κυριως', 'λιγακι', 'λιγο',
			  'λιγοτερο', 'λογω', 'λοιπα', 'λοιπον', 'μα', 'μαζι', 'μακαρι', 'μακρια', 'μαλιστα', 'μαλλον', 'μας',
			  'με', 'μεθαυριο', 'μειον', 'μελει', 'μελλεται', 'μεμιας', 'μεν', 'μερικα', 'μερικες', 'μερικοι',
			  'μερικους', 'μερικων', 'μεσα', 'μετ', 'μετα', 'μεταξυ', 'μεχρι', 'μη', 'μηδε', 'μην', 'μηπως', 'μητε',
			  'μια', 'μιαν', 'μιας', 'μολις', 'μολονοτι', 'μοναχα', 'μονες', 'μονη', 'μονη', 'μονης', 'μονο',
			  'μονοι', 'μονομιας', 'μονος', 'μονου', 'μονους', 'μονων', 'μου', 'μπορει', 'μπορουν', 'μπραβο',
			  'μπρος', 'να', 'ναι', 'νωρις', 'ξανα', 'ξαφνικα', 'ο', 'οε', 'οι', 'ολα', 'ολες', 'ολη', 'ολην',
			  'ολης', 'ολο', 'ολογυρα', 'ολοι', 'ολον', 'ολος', 'ολοτελα', 'ολου', 'ολους', 'ολων', 'ολως',
			  'ολωσδιολου', 'ομως', 'οποια', 'οποιαδηποτε', 'οποιαν', 'οποιανδηποτε', 'οποιας', 'οποιοδηποτε',
			  'οποιες', 'οποιεσδηποτε', 'οποιο', 'οποιοδηποτε', 'οποιοι', 'οποιον', 'οποιονδηποτε', 'οποιος',
			  'οποιοσδηποτε', 'οποιου', 'οποιουδηποτε', 'οποιους', 'οποιων', 'οποιωνδηποτε', 'οποτε', 'οποτεδηποτε',
			  'οπου', 'οπουδηποτε', 'οπως', 'ορισμενα', 'ορισμενες', 'ορισμενων', 'ορισμενως', 'οσα', 'οσαδηποτε',
			  'οσες', 'οσεσδηποτε', 'οση', 'οσηδηποτε', 'οσην', 'οσηνδηποτε', 'οσης', 'οσησδηποτε', 'οσο',
			  'οσοδηποτε', 'οσοι', 'οσοιδηποτε', 'οσον', 'οσονδηποτε', 'οσος', 'οσοσδηποτε', 'οσου', 'οσουδηποτε',
			  'οσους', 'οσουσδηποτε', 'οσων', 'οσωνδηποτε', 'οταν', 'οτι', 'οτιδηποτε', 'οτου', 'ου', 'ουδε',
			  'ουτε', 'παλι', 'παντοτε', 'παντου', 'παντως', 'παρα', 'περα', 'περι', 'περιπου', 'περισσοτερο',
			  'περσι', 'περυσι', 'πια', 'πιθανον', 'πιο', 'πισω', 'πλαι', 'πλεον', 'πλην', 'ποια', 'ποιαν', 'ποιας',
			  'ποιες', 'ποιο', 'ποιοι', 'ποιον', 'ποιος', 'ποιου', 'ποιους', 'ποιων', 'πολυ', 'ποσες', 'ποση',
			  'ποσην', 'ποσης', 'ποσοι', 'ποσος', 'ποσους', 'ποτε', 'που', 'πουθε', 'πουθενα', 'πρεπει', 'πριν',
			  'προ', 'προκειμενου', 'προκειται', 'προπερσι', 'προς', 'προτου', 'προχθες', 'προχτες', 'πρωτυτερα',
			  'πως', 'σαν', 'σας', 'σε', 'σεις', 'σημερα', 'σιγα', 'σου', 'στα', 'στη', 'στην', 'στης', 'στις',
			  'στο', 'στον', 'στου', 'στους', 'στων', 'συγχρονως', 'συν', 'συναμα', 'συνεπως', 'συνηθως', 'συχνα',
			  'συχνες', 'συχνη', 'συχνην', 'συχνης', 'συχνο', 'συχνοι', 'συχνον', 'συχνος', 'συχνου', 'συχνους',
			  'συχνων', 'σχεδον', 'σωστα', 'τα', 'ταδε', 'ταυτα', 'ταυτες', 'ταυτη', 'ταυτην', 'ταυτης', 'ταυτο',
			  'ταυτον', 'ταυτος', 'ταυτου', 'ταυτων', 'ταχα', 'ταχατε', 'τελικα', 'τελικως', 'τες', 'τετοια',
			  'τετοιαν', 'τετοιας', 'τετοιες', 'τετοιο', 'τετοιοι', 'τετοιον', 'τετοιος', 'τετοιου', 'τετοιους',
			  'τετοιων', 'τη', 'την', 'της', 'τι', 'τιποτα', 'τιποτε', 'τις', 'το', 'τοι', 'τον', 'τος', 'τοσα',
			  'τοσες', 'τοση', 'τοσην', 'τοσης', 'τοσο', 'τοσοι', 'τοσον', 'τοσος', 'τοσου', 'τοσους', 'τοσων',
			  'τοτε', 'του', 'τουλαχιστο', 'τουλαχιστον', 'τους', 'τουτα', 'τουτες', 'τουτη', 'τουτην', 'τουτης',
			  'τουτο', 'τουτοι', 'τουτοις', 'τουτον', 'τουτος', 'τουτου', 'τουτους', 'τουτων', 'τυχον', 'των',
			  'τωρα', 'υπ’', 'υπερ', 'υπο', 'υποψη', 'υποψιν', 'υστερα', 'φετος', 'χαμηλα', 'χθες', 'χτες', 'χωρις',
			  'χωριστα', 'ψηλα', 'ω', 'ωραια', 'ως', 'ωσαν', 'ωσοτου', 'ωσπου', 'ωστε', 'ωστοσο', 'ωχ', '-', '+',
			  '=', '«', '»', '(', ')', '[', ']', ';', '?', '<', '> ', 'βιντεο', 'video', 'vid', 'RT', 'retweet']
tags = ['!', '"', '$', '%', '&', '\'', '(', ')', '*', '+', ',', '-', '.', '/', ':', ';', '<', '=', '>', '?', '[',
		']', '^', '_', '`', '{', '|', '}', '~', '«', '»', '€']

Tfidf_vect = pickle.load(open("vectorizer.pickle", "rb"))

def word_spliter(page_text):

	global GREEK_STOP, tags


	page_text = page_text.encode('utf-8')
	page_text = page_text.lower()
	for tag in tags:
		page_text=page_text.replace(tag, '')
	page_text = ', '.join(word_tokenize(page_text))

	# Afairesh tonwn
	def strip_accents(s):
		return ''.join(c for c in unicodedata.normalize('NFD', s)
					   if unicodedata.category(c) != 'Mn')

	page_text = strip_accents(page_text.decode('utf-8'))

	page_text = page_text.split(",")
	final_words = []
	for word in page_text:
		if word.encode('utf-8') not in GREEK_STOP:
			final_words.append(word)
	return page_text

def prediction_NB(page):

	global Tfidf_vect

	Naive = pickle.load(open('NB_model.sav', 'rb'))
	TestX=Tfidf_vect.transform(page)
	predictions_NB = Naive.predict(TestX)
	predictions_NB_proba = Naive.predict_proba(TestX)
	proba = predictions_NB_proba[:, 0]
	proba = np.mean(proba)

	return proba

def prediction_SVM(page):

	global Tfidf_vect

	SVM = pickle.load(open('SVM_model.sav', 'rb'))
	TestX = Tfidf_vect.transform(page)
	predictions_SVM = SVM.predict(TestX)
	predictions_SVM_proba = SVM.predict_proba(TestX)
	proba = predictions_SVM_proba[:, 0]
	proba = np.mean(proba)

	return proba

def focused_crawler(seed_url):

	URL_ranking = dict()
	checked=[]

	crawled_count = 0

	frontier_urls = [seed_url]
	frontier_url = seed_url
	URL_ranking.update( {seed_url : 0.0} )
	seen_urls = []

	newpath = r'Logs'
	if not os.path.exists(newpath):
		os.makedirs(newpath)

	# if not os.path.exists(newpath):
	# 	os.makedirs(newpath)
	crawler_log = open("Logs/crawler_log.txt","w")
	crawler_log.write("Seed : "+seed_url+"\n\n")

	crawled_count+=1
	crawler_log.write(str(crawled_count)+") "+seed_url+"\n\n")

	# flag is True if the limit of 1000 URLS has not been reached
	flag = True
	print(str(crawled_count)+") "+seed_url)

	source_code = requests.get(seed_url)
	plain_text = source_code.text
	soup = BeautifulSoup(plain_text,"html.parser")
	#ama 8eloume na to katebasoume ginetai apo dw

	# name = seed_url[seed_url.rfind('/')+1:]
	# name = open("Raw_HTML_Downloads/"+(str(crawled_count))+") "+name+".txt","w")
	# name.write(str(soup.prettify(encoding='utf-8')))
	# name.close()

	while(flag):

		extracted_urls = []

			# Traversing through all the URLs to be crawled as pointed by the Frontier
		while (frontier_url is not None):

			# Enter only if limit of 1000 URLs not reached
			if flag:

				time.sleep(1)

				# get the soup
				source_code = requests.get(frontier_url)
				if source_code == None: continue
				plain_text = source_code.text
				soup = BeautifulSoup(plain_text,"html.parser")

				url = frontier_url

				extracted_urls.append(url) #to bazoume sta katebasmena

				crawled_count += 1
				crawler_log.write(str(crawled_count) + ") " + url + "\n")
					
				try:
					URL_ranking[url]
				except Exception:
					checked.append(frontierbyValue[0])
					frontierbyValue.pop(0)
					frontier_url=frontierbyValue[0][0]
					continue
					
				print(str(crawled_count) + ") " + url+' ' + str(URL_ranking[url]))

				URL_ranking.pop(url)  # bgale to url apo th lista
				page = soup.find_all('p')
				content = ' '.join(item.text for item in page)

				new_page=content#str(content.encode('utf-8'))
				page = word_spliter(new_page)
#			#choose Naive Bayes or SVM to rate
				rate = prediction_NB(page)
				#rate = prediction_SVM(page)

				for link in soup.find_all('a', href=re.compile('')):

					if not link['href'].startswith(seed_url):
						continue

					if not link['href'].endswith('.html'):
						continue

					if crawled_count < 1000 and flag:

						href_url = link['href']

						if ':' in href_url:

							url = href_url

							# URL should not be in either of Frontier, Extracted or Seen lists and should not be Wiki Main Page too
							if url not in frontier_urls and url not in extracted_urls and url not in seen_urls:

								URL_ranking.update({url:rate})
								frontierbyValue = sorted(URL_ranking.items(), reverse=True, key=lambda x: x[1])
								frontier_url=frontierbyValue[0][0]

								if len(frontierbyValue)>200:
									frontierbyValue = frontierbyValue[:len(frontierbyValue) // 2] #pare th prwth mish
									URL_ranking=dict(frontierbyValue) # meiwse sth mesh to le3iko
							else:
								frontier_url = frontierbyValue[1][0]

				seen_urls.append(extracted_urls)

		if len(extracted_urls) == 0:
			flag = False
			break
		frontier_urls = extracted_urls
		crawler_log.write("\n")

	if flag:
		print("Searched")

	crawler_log.write("--------\n")
	crawler_log.write("Logistics :\n\n")
	crawler_log.write("Number of matching searches : "+str(crawled_count)+"\n")

seed_url = 'https://www.sport24.gr'
focused_crawler(seed_url)
