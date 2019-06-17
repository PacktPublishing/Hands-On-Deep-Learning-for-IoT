# -*- coding: utf-8 -*-
"""
Created on Wed May  8 13:27:41 2019
This is the code for scrapping imags of skin diseses from a given link.
@author: Raz
"""

# Imports the modules
import requests
from bs4 import BeautifulSoup
import os
import shutil

# Root URL 

root_URL = 'http://www.dermnet.com/'

# Skin diseases types 
type_name = ['Acne-and-Rosacea-Photos', 'Actinic-Keratosis-Basal-Cell-Carcinoma-and-other-Malignant-Lesions', 
             'Atopic-Dermatitis-Photos', 'Bullous-Disease-Photos', 'Cellulitis-Impetigo-and-other-Bacterial-Infections', 
             'Eczema-Photos', 'Exanthems-and-Drug-Eruptions', 'Hair-Loss-Photos-Alopecia-and-other-Hair-Diseases',
             'Herpes-HPV-and-other-STDs-Photos', 'Light-Diseases-and-Disorders-of-Pigmentation',
             'Lupus-and-other-Connective-Tissue-diseases', 'Melanoma-Skin-Cancer-Nevi-and-Moles', 'Nail-Fungus-and-other-Nail-Disease', 
             'Poison-Ivy-Photos-and-other-Contact-Dermatitis', 'Psoriasis-pictures-Lichen-Planus-and-related-diseases', 
             'Scabies-Lyme-Disease-and-other-Infestations-and-Bites', 'Seborrheic-Keratoses-and-other-Benign-Tumors', 
             'Systemic-Disease', 'Tinea-Ringworm-Candidiasis-and-other-Fungal-Infections', 
             'Urticaria-Hives', 'Vascular-Tumors', 'Vasculitis-Photos', 'Warts-Molluscum-and-other-Viral-Infections']


type_LinksA = []
type_PagesA = []
type_SubLinksA = []

def find_max(link_):
    r_ = requests.get(link_)
    html_page1_1 = r_.text
    soup_page1_1 = BeautifulSoup(html_page1_1, "html5lib")

    navigationL = soup_page1_1.find_all("div", attrs={"class": "pagination"})
    max_ = 1
    last_ =1

    if not navigationL:
        pass
    else:
        for navi_ in navigationL: 
            for i in navi_.children:
                try:
                    i_ = i.contents[0]
                except:
                    pass
                else:
                    if i_ == 'Next':
                        max_ = int(last_)
                        break
                    else: 
                        last_ = i_
    return int(max_)

# Function to convert images to downloadble link for the images

def images2links(imagesL):
    thumbRLinks = []
    
    for url_ in imagesL:
        
        soup_page1_1_ = BeautifulSoup(requests.get(url_).text, "html5lib")

        for link in soup_page1_1_.find_all("img"):
            link_ = link.get("src")
            if 'Thumb' in link_:
                thumbRLinks.append(link_.replace('Thumb', ''))
    return list(set(thumbRLinks))

def download(url, image_path):
    try:
        r = requests.get(url, stream=True)
        if r.status_code == 200:
            with open(image_path, 'wb') as f:
                r.raw.decode_content = True
                shutil.copyfileobj(r.raw, f)
    except Exception as e:
        print (e)
        print ("Failed to save " + image_path)
        print (url + "\n")

    else:
        print ("Successfully saved " + image_path)


# Main function 
def main():
    for sub_ in type_name:
        SubLinks = []        
        r_ = requests.get(root_URL + '/images/' + sub_)
        html_page1 = r_.text
        soup_page1 = BeautifulSoup(html_page1, "html5lib")

        for link in soup_page1.find_all('a'):
            id_ = link.get('href')
            if '/images/' in id_:
                
                SubLinks.append(root_URL + id_)
        type_LinksA.append(SubLinks)

    attr = '/photos/'

    for type_ in type_LinksA:
        maxL_ = []
        thumbRLinks = []
        
        for link_ in type_:
            imagesL = []
            max_ = find_max(link_)
            maxL_.append(max_)
            print (link_)
            for i_ in range(max_):
                imagesL.append(link_ + attr + str(i_+1))
            
            print ("Pages: ", len(imagesL))
            realLinks = images2links(imagesL)
            print ("Links: ", len(realLinks))
            thumbRLinks.append(realLinks)

        type_PagesA.append(maxL_)
        type_SubLinksA.append(thumbRLinks)

    dir_root_stored_images = './DermnetDataset/'

    if os.path.exists(dir_root_stored_images):
        pass    
    else:
        os.mkdir(dir_root_stored_images) 
    for category in type_name:
        iloc = type_name.index(category)
        dir_disease = dir_root_stored_images + category + '/'
        
        if os.path.exists(dir_disease):
            pass    
        else:
            os.mkdir(dir_disease) 
            
        print ("Disease: ", dir_disease)
        for sub_ in type_LinksA[iloc]:
            iloc_sub = type_LinksA[iloc].index(sub_)
           
            dir_sub_disease = dir_disease + sub_.split('/')[-1]
            
            
            if os.path.exists(dir_sub_disease):
                pass    
            else:
                os.mkdir(dir_sub_disease)
            
            for l_ in type_SubLinksA[iloc][iloc_sub]:
                img_path = dir_sub_disease + '/' + l_.split('/')[-1]
                img_url = l_
                download(img_url, img_path)

if __name__ == "__main__":
    main()
