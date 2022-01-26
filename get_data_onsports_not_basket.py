# -*- coding: utf-8 -*-

from bs4 import BeautifulSoup
import requests
import json
import numpy as np
import pandas as pd

df = pd.DataFrame(columns = ('url', 'title', 'content' ))

def crawling(url, category):
    count = 0
    result = []
    req = requests.get(url)
    soup = BeautifulSoup(req.text, "lxml")

    news_links=[]

#    df = pd.DataFrame(columns = ('url', 'title', 'content' ))
    global df

    #looping through paging
    for i in range(0, 91, 15):
        url_paged = url+ "/" + category + "?start=" + str(i)

        #find article link
        req = requests.get(url_paged+str(i))
        soup = BeautifulSoup(req.text, "lxml")
        a = soup.find("div",{'class':'story-block md33 has-img list-story-0'})
        
        for j in range(0,10):
            news_links.append(soup.find("div",{'class':'story-block md33 has-img list-story-'+str(j)}))

        #looping through article link
        for idx,news in enumerate(news_links):

            #find news title
            title_news= news.find('h3',{'class':'story-title'}).text.strip('\n').strip('\t')

            #find urll news
            url_news = news.find('a',{'class':'story-link'}).get('href')

            #find news content in url
            req_news =  requests.get(url+url_news+'.gr')
            soup_news = BeautifulSoup(req_news.text, "lxml")

            #find news content 
            news_intro_content = soup_news.find("div",{'class':'story-intro'})

            #find news content 
            news_content = soup_news.find("div",{'class':'story-text story-fulltext'})

            if (news_content == None) and (news_intro_content == None) :
                continue

            count=count+1
            if count == 57: break
            print count

            #find paragraph in news content
            p = news_content.find_all('p')
            intro_content = news_intro_content.find('p').text
            intro_content = intro_content.encode('utf8','replace')
            content = ' '.join(item .text for item in p)
            news_content = content.encode('utf8','replace')
            news_content = intro_content + ' ' + news_content

            df.loc[len(df)] = [url+url_news+'.gr', title_news, news_content]

    #print df
         
    #return df

url = 'https://www.onsports.gr'
#crawl  = crawling(url, 'podosfairo')
crawl  = crawling(url, 'spor')
crawl  = crawling(url, 'auto-moto')
crawl  = crawling(url, 'be-a-pro')
crawl  = crawling(url, 'hot')
crawl  = crawling(url, 'perierga')
df.to_csv('onsports_not_basket.csv', encoding = 'utf-8')

