#!/usr/bin/env python
# coding: utf-8

# In[1]:


# Importing Libraries
import requests
from bs4 import BeautifulSoup
import pandas as pd

# List comprehension function to scrape webpage titles
# Stores everything in memory and produces all results at once.
def list_comprehension(url, tag, **kwargs):
    try:
        response = requests.get(url)
        response.raise_for_status()  # Raise a HTTP Error for bad responses
        soup = BeautifulSoup(response.content, 'html.parser')
        
        class_name = kwargs.get('class_name')
        
        if class_name:
            titles = [title.text for title in soup.find_all(tag) if title.find('span', class_=class_name)]
        else:
            titles = [title.text for title in soup.find_all(tag)]
        
        return titles[:10]
    except requests.RequestException as e:
        print(f"Failed to retrieve webpage: {e}")
        return []

# Generator function to scrape webpage titles
# A generator processes one URL at a time.
# Generator function to Scrape Webpage Titles
def generator(url, tag, **kwargs):
    count = 0
    try:
        response = requests.get(url)
        response.raise_for_status()  # Raise an HTTP Error for bad responses
        soup = BeautifulSoup(response.content, 'html.parser')

        class_name = kwargs.get('class_name')

        for title in soup.find_all(tag):
            if count >= 10:
                return
            if class_name and title.find('span', class_=class_name):
                yield title.text
                count += 1
            elif not class_name:
                yield title.text
                count += 1

    except requests.RequestException as e:
        print(f"Failed to retrieve webpage: {e}")


# In[6]:


# urls
urls = ['https://www.rte.ie/', 'https://www.bbc.com/news']


# In[7]:


# LIST COMPREHENSION

# Call scrape_titles function on RTE and BBC websites
rte_titles = list_comprehension('https://www.rte.ie/', 'h3', class_name="underline")
bbc_titles = list_comprehension('https://www.bbc.com/news', 'h3') 
print(rte_titles)

# Convert titles to dataframes with a common index
rte_df = pd.DataFrame(rte_titles, columns=['RTE Titles'])
bbc_df = pd.DataFrame(bbc_titles, columns=['BBC Titles'])
print(rte_df)

# Merge dataframes using pd.concat() based on index  # Merging Dataframes
merged_df = pd.concat([rte_df, bbc_df], axis=1) 

# Define the columns we'll be processing
base_columns = ['RTE Titles', 'BBC Titles']

# Data Cleaning and Processing
for col in base_columns:
    # Cleaning Up White Spaces
    merged_df[col] = merged_df[col].str.replace(r'\s+', ' ', regex=True).str.strip()
    
    # Removing Special Characters
    merged_df[f'{col} Cleaned'] = merged_df[col].str.replace(r'[!?.]', '', regex=True)
    
    # Highlighting if the Titles Contain Numbers
    merged_df[f'{col} Contains Numbers'] = merged_df[col].str.contains(r'\d', regex=True)
    
    # Identifying if the Titles Mention "UCD"
    merged_df[f'{col} Mentions UCD'] = merged_df[col].str.contains(r'UCD', regex=True)

# Display the processed dataframe
print(merged_df)


# In[8]:


# GENERATOR FUNCTION
# Using the generator for each URL
rte_titles = list(generator(urls[0], 'h3', class_name="underline"))
bbc_titles = list(generator(urls[1], 'h3'))

# Convert titles to dataframes with a common index
rte_df = pd.DataFrame(rte_titles, columns=['RTE Titles'])
bbc_df = pd.DataFrame(bbc_titles, columns=['BBC Titles'])

# Merge dataframes using pd.concat() based on index
merged_df = pd.concat([rte_df, bbc_df], axis=1) 

# Define the columns we'll be processing
base_columns = ['RTE Titles', 'BBC Titles']

# Data Cleaning and Processing
for col in base_columns:
    # Cleaning Up White Spaces
    merged_df[col] = merged_df[col].str.replace(r'\s+', ' ', regex=True).str.strip()
    
    # Removing Special Characters
    merged_df[f'{col} Cleaned'] = merged_df[col].str.replace(r'[!?.]', '', regex=True)
    
    # Highlighting if the Titles Contain Numbers
    merged_df[f'{col} Contains Numbers'] = merged_df[col].str.contains(r'\d', regex=True)
    
    # Identifying if the Titles Mention "UCD"
    merged_df[f'{col} Mentions UCD'] = merged_df[col].str.contains(r'UCD', regex=True)

# Display the processed dataframe
print(merged_df)


# In[9]:


# Both of these techniques (generator and list comprehension) provide 
# a way to efficiently handle multiple URLs or data processing steps 
# without having to manually loop and append results. 

# If you want to lazily process one item at a time, use a generator.
# If you want all results at once, use a list comprehension.

# The list comprehension is the best in this scenario as it is a little more simple 
# and since the information being stored is quite small, it won't have a
# big effect on performance.

