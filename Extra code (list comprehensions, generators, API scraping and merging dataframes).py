# Importing Libraries
import requests
from bs4 import BeautifulSoup
import pandas as pd

# Define Function to Scrape Webpage Titles
def scrape_titles(url, tag, **kwargs):
    try:
        response = requests.get(url)
        response.raise_for_status()  # Raise an HTTPError for bad responses
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

# Call scrape_titles function on RTE and BBC websites
rte_titles = scrape_titles('https://www.rte.ie/', 'h3', class_name="underline")
bbc_titles = scrape_titles('https://www.bbc.com/news', 'h3') 

# Convert titles to dataframes with a common index
rte_df = pd.DataFrame(rte_titles, columns=['RTE Titles'])
bbc_df = pd.DataFrame(bbc_titles, columns=['BBC Titles'])

# Use pd.merge() to merge on the index
merged_df = pd.merge(rte_df, bbc_df, left_index=True, right_index=True)
print(merged_df)

# # Merge dataframes
# merged_df = pd.concat([rte_df, bbc_df], axis=1) 

base_columns = ['RTE Titles', 'BBC Titles']

for col in base_columns:
    # Cleaning Up White Spaces
    merged_df[col] = merged_df[col].str.replace(r'\s+', ' ', regex=True).str.strip()
    
    # Highlighting Numbers
    merged_df[f'{col} Contains Numbers'] = merged_df[col].str.contains(r'\d', regex=True)
    
    # Removing Special Characters
    merged_df[f'{col} Cleaned'] = merged_df[col].str.replace(r'[!?.]', '', regex=True)
    
    # Identifying Keywords (for example, headlines that contain "COVID-19")
    merged_df[f'{col} /Mentions UCD'] = merged_df[col].str.contains(r'UCD', regex=True)

print(merged_df)
