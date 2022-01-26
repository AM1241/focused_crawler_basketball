# -*- coding: utf-8 -*-

from bs4 import BeautifulSoup
import requests
import json
import numpy as np
import pandas as pd

df = pd.DataFrame(columns = ('url', 'title', 'content' ))

def crawling(url, category):
    
    result = []
    req = requests.get(url)
    soup = BeautifulSoup(req.text, "lxml")

    news_links=[]

#    df = pd.DataFrame(columns = ('url', 'title', 'content' ))
    global df
    count = 0

    #looping through paging
    for i in range(2, 20, 1):
        url_paged = url +"/category/"+ category + "/page/" + str(i) + "/"

        #find article link
        req = requests.get(url_paged)
        soup = BeautifulSoup(req.text, "lxml")
        #a = soup.find("div",{'class':'story-block md33 has-img list-story-0'})
        a = soup.find("div",{'class':'td_module_wrap prel mtasda2'})
        
        for j in range(2,30):
            #news_links.append(soup.find("div",{'class':'story-block md33 has-img list-story-'+str(j)}))
            news_links.append(soup.find("div",{'class':'td_module_wrap prel mtasda'+str(j)}))


        #looping through article link
        for idx,news in enumerate(news_links):

            #find news title
            #title_news= news.find('h3',{'class':'story-title'}).text.strip('\n').strip('\t')
            title_news= news.find('a').get('title')

            #find urll news
            url_news = news.find('a').get('href')

            #find news content in url
            req_news =  requests.get(url_news)
            soup_news = BeautifulSoup(req_news.text, "lxml")

            #find news content 
            news_content = soup_news.find("div",{'class':'postcontent'})
            if news_content == None :
                continue

            #find news content 
            #news_content = soup_news.find("div",{'class':'story-text story-fulltext'})

            #find paragraph in news content
            p = news_content.find_all('p')
            #intro_content = news_intro_content.find('p').text
            #intro_content = intro_content.encode('utf8','replace')
            content = ' '.join(item .text for item in p)
            news_content = content.encode('utf8','replace')
            #news_content = intro_content + ' ' + news_content

            df.loc[len(df)] = [url_news, title_news, news_content]
            #print [url_news, title_news, news_content]
            count = count + 1
            if count%100==0:
                print count
            if count == 600: break
        if count == 600: break


    #print df
         
    return df

url = 'https://www.to10.gr'
crawl  = crawling(url, 'podosfero')
crawl  = crawling(url, 'sport')
#crawl  = crawling(url, 'cars-n-bikes')
crawl  = crawling(url, 'ten-files')
crawl.to_csv('to10_not_basket.csv', encoding = 'utf-8')



