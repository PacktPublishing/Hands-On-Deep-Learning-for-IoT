#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Mar 31 23:03:13 2019

@author: raz
"""
## Import the required modules

import urllib
from bs4 import BeautifulSoup
from selenium import webdriver
import os, os.path
import simplejson

from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC

## Create links book for the audio data to be downloaded: this may include repeated readers

book_links = []

browser = webdriver.PhantomJS(executable_path = '/usr/local/bin/phantomjs')


for i in range(1): ## testing first 0-1 (2) pages of the site : to minimise the time require to downloads
    
    url = ("https://librivox.org/search?title=&author=&reader=&keywords=&genre_id=0&status=all&project_type=solo&recorded_language=&sort_order=catalog_date&search_page={}&search_form=advanced").format(i)
    
    print(url)
    
    browser.get(url)
    element = WebDriverWait(browser, 100).until(
    EC.presence_of_element_located((By.CLASS_NAME , "catalog-result")))
    html = browser.page_source
    soup = BeautifulSoup(html, 'html.parser')
    
    ul_tag = soup.find('ul', {'class': 'browse-list'})
    
    for li_tag in ul_tag.find_all('li', {'class': 'catalog-result'}):
        result_data = li_tag.find('div', {'class': 'result-data'})
        book_meta = result_data.find('p', {'class': 'book-meta'})
        link = result_data.a["href"]
        print(link)
        if str(book_meta).find("Complete") and link not in book_links:
            book_links.append(link)
            
    print(len(book_links)) # links per page could be different from regular browsers
    
browser.quit()

## save the gathered or scrapped links to a folder (audio_book_links_speakers.txt)

f = open('audio_book_links_speakers.txt', 'w')
simplejson.dump(book_links, f)
f.close()


## Extract the potential reader or speaker list

browser = webdriver.PhantomJS(executable_path = '/usr/local/bin/phantomjs')

reader_list = []
download_links = []
download_sizes = []

repreader = 0

for i in range(len(book_links)):
    
    link = book_links[i]
    
    browser.get(link)
    html = browser.page_source
    soup = BeautifulSoup(html, 'html.parser')
    
    product_details = soup.find('dl', {'class': 'product-details clearfix'})
    
    if product_details == None:
        continue
    
    product_details_list = product_details.find_all("dd")
    
    reader = product_details_list[3].get_text()
    size_mb = product_details_list[1].get_text()
    
    try:
        size_mb = float(size_mb.replace('MB',''))
    except:
        continue
    
    if reader not in reader_list:
            
        reader_list.append(reader)
        download_sizes.append(size_mb)

        listen_download = soup.find('dl', {'class': 'listen-download clearfix'})
        zip_download = listen_download.a["href"]

        print(reader, str(len(reader_list)) + "/" + str(i+1) + " potentials")

        download_links.append(zip_download)
            
    else:
        repreader += 1
        print("repeat reader " + str(repreader))
            
        
browser.quit()

### The downloads the for non repeatable readers/speakers

#  List of Links or pages for the audio books to be downloaded
f = open('audiodownload_links.txt', 'w')
simplejson.dump(download_links, f)
f.close()

## Record the file size of each reader's file

f = open('audiodownload_sizes.txt', 'w')
simplejson.dump(download_sizes, f)
f.close()


#### Download the files 
#
#
def count_files():
    dir = 'audio_files_downloaded'
    list = [file for file in os.listdir(dir) if file.endswith('.zip')] # dir is your directory path
    number_files = len(list)
    return number_files
#
counter = 10 # this is to use for nameing the individual downloaded file
#
###
#    
for link, size in zip(download_links, download_sizes):
    if size >= 50 and size <= 100:
        localDestination = 'audio_files_downloaded/audio{}.zip'.format(counter)
        resultFilePath, responseHeaders = urllib.request.urlretrieve(link, localDestination)
        counter += 1
#
cnt2 =  0
num = count_files()

if num < 200:
    for link, size in zip(download_links, download_sizes):
        if size > 100 and size <= 150:
            localDestination = 'audio_files_downloaded/audio{}.zip'.format(counter)
            resultFilePath, responseHeaders = urllib.request.urlretrieve(link, localDestination)
            counter += 1
        cnt2 += 1

num = count_files()

if num < 200:
    for link, size in zip(download_links, download_sizes):
        if size > 150 and size <= 200:
            localDestination = 'audio_files_downloaded/audio{}.zip'.format(counter)
            resultFilePath, responseHeaders = urllib.request.urlretrieve(link, localDestination)
            counter += 1
        
num = count_files()

if num < 200:
    for link, size in zip(download_links, download_sizes):
        if size > 200 and size <= 250:
            localDestination = 'audio_files_downloaded/audio{}.zip'.format(counter)
            resultFilePath, responseHeaders = urllib.request.urlretrieve(link, localDestination)
            counter += 1
        
num = count_files()

if num < 200:
    for link, size in zip(download_links, download_sizes):
        if size > 250 and size <= 300:
            localDestination = 'audio_files_downloaded/audio{}.zip'.format(counter)
            resultFilePath, responseHeaders = urllib.request.urlretrieve(link, localDestination)
            counter += 1
            
num = count_files()

if num < 200:
    for link, size in zip(download_links, download_sizes):
        if size > 300 and size <= 350:
            localDestination = 'audio_files_downloaded/audio{}.zip'.format(counter)
            resultFilePath, responseHeaders = urllib.request.urlretrieve(link, localDestination)
            counter += 1
            
num = count_files()

if num < 200:
    for link, size in zip(download_links, download_sizes):
        if size > 350 and size <= 400:
            localDestination = 'audio_files_downloaded/audio{}.zip'.format(counter)
            resultFilePath, responseHeaders = urllib.request.urlretrieve(link, localDestination)
            counter += 1
# 
#            
