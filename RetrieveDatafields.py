#!/usr/bin/env python
# coding: utf-8

# In[2]:


import requests
from dotenv import load_dotenv
import os
from urllib.parse import urljoin
from time import sleep
import json
from datetime import datetime
import pandas as pd
load_dotenv()
import time
import ast
import json

brain_api_url = os.environ.get("BRAIN_API_URL", "https://api.worldquantbrain.com")
brain_url = os.environ.get("BRAIN_URL", "https://platform.worldquantbrain.com") 


class BRAINAPIWRAPPER:
    
    def __init__(self, id_file='retrieved_ids.json'):
        self.session = self.get_login_session()
        self.permissions = self.check_permissions()  # Initialize permissions here
        self.id_file = id_file
        self.retrieved_ids = self.load_retrieved_ids()  # Load existing IDs from the file

        
    def get_login_session(self):
        session = requests.Session()
        username = os.getenv('wqbrain_consultant_user')
        password = os.getenv('wqbrain_consultant_pw')
        session.auth = (username, password)
        response = session.post('https://api.worldquantbrain.com/authentication')
        response.headers
        print(username)

        if response.status_code == requests.status_codes.codes.unauthorized:
            if response.headers["WWW-Authenticate"] == "persona":
                biometric_url = urljoin(response.url, response.headers["Location"])
                print(biometric_url)
                input("Complete bio" + biometric_url)
                biometric_response = session.post(biometric_url)
        else:
            print("incorrect")
        return session
    
    def check_permissions(self):
        response = self.session.get('https://api.worldquantbrain.com/authentication')
        
        if response.status_code == 200:
            data = response.json()
            permissions = data.get('permissions', [])
            print("User permissions: ", permissions)
            return permissions
        else:
            print("Failed to retrieve permissions: ", response.status_code)
            return []
    
    def has_multi_simulation_permission(self):
        return "MULTI_SIMULATION" in self.permissions
    


# In[3]:


def load_retrieved_ids(self):
    """Load retrieved IDs from a JSON file."""
    if os.path.exists(self.id_file):
        with open(self.id_file, 'r') as f:
            return set(json.load(f))  # Load IDs into a set for fast lookup
    return set()  # Return an empty set if the file does not exist

BRAINAPIWRAPPER.load_retrieved_ids = load_retrieved_ids


def save_retrieved_ids(self):
    """Save retrieved IDs to a JSON file."""
    with open(self.id_file, 'w') as f:
        json.dump(list(self.retrieved_ids), f)  # Save IDs as a list

BRAINAPIWRAPPER.save_retrieved_ids = save_retrieved_ids


# In[4]:


s = BRAINAPIWRAPPER()


# In[4]:


def retrieve_datafields2(
    self,
    instrument_type: str = 'EQUITY',
    region: str = 'GLB',
    delay: int = 1,
    universe: str = 'MINVOL1M',
    dataset_id: str = '',
    search: str = '',
    batch_size: int = None,  # Remove the default value
    max_volume: int = 50   # New parameter to specify the maximum volume
):
    # Set a default value for batch_size if it is not provided
    if batch_size is None:
        batch_size = 10  # You can set any default value you prefer

    offset = 0
    total_count = 0
    all_results = []

    while True:
        # Construct the URL for the API request
        url_template = brain_api_url + "/data-fields?" +\
            f"&instrumentType={instrument_type}" +\
            f"&region={region}&delay={str(delay)}&universe={universe}&limit={batch_size}" +\
            f"&offset={offset}"

        # Attempt to retrieve data
        for _ in range(5):  # Retry up to 5 times
            response = self.session.get(url_template)
            data = response.json()
            if 'results' in data:
                break
            else:
                time.sleep(5)

        # Check if there are results
        results = data.get('results', [])
        if not results:
            print("No results.")
            break  # Exit the loop if no more results

        # Highlighted Change: Print the retrieved results for this batch
        print(f"Retrieved batch of {len(results)} datafields:")
        new_results = []  # List to store new results that are not already retrieved

        for datafield in results:
            # Assuming each datafield has a unique identifier, e.g., 'id'
            datafield_id = datafield.get('id')  # Adjust this based on your data structure

            # Check if this datafield has already been retrieved
            if datafield_id not in self.retrieved_ids:
                new_results.append(datafield)  # Add to new results
                self.retrieved_ids.add(datafield_id)  # Mark as retrieved

        # If there are new results, extend the all_results list
        if new_results:
            all_results.extend(new_results)
            total_count += len(new_results)

            # Print each new data field
            for index, datafield in enumerate(new_results):
                print(f"Datafield {index + 1}: {datafield}")

        # Check if total_count has reached or exceeded max_volume
        if total_count >= max_volume:
            print(f"Reached maximum volume of {max_volume} datafields.")
            break  # Exit the loop if the maximum volume is reached


        
        # Optional: Print progress
        print(f'Total datafields retrieved so far: {total_count}')

    # Save the retrieved IDs to the file
    self.save_retrieved_ids()
    print(f'Total datafields retrieved: {total_count}')
    return all_results  # Return all collected results


BRAINAPIWRAPPER.retrieve_datafields2 = retrieve_datafields2


# In[6]:


def retrieve_datafields3(
    self,
    instrument_type: str = 'EQUITY',
    region: str = 'GLB',
    delay: int = 1,
    universe: str = 'MINVOL1M',
    dataset_id: str = '',
    search: str = '',
    batch_size: int = None,
    max_volume: int = 50
):
    if batch_size is None:
        batch_size = 50

    offset = 0
    total_count = 0
    all_results = []
    no_new_data_count = 0  # Counter for no new data

    while True:
        url_template = brain_api_url + "/data-fields?" +\
            f"&instrumentType={instrument_type}" +\
            f"&region={region}&delay={str(delay)}&universe={universe}&limit={batch_size}" +\
            f"&offset={offset}"

        for _ in range(5):
            response = self.session.get(url_template)
            data = response.json()
            if 'results' in data:
                break
            else:
                time.sleep(5)

        results = data.get('results', [])
        if not results:
            print("No results.")
            break

#        print(f"Retrieved batch of {len(results)} datafields:")
        new_results = []

        for datafield in results:
            datafield_id = datafield.get('id')
            if datafield_id not in self.retrieved_ids:
                new_results.append(datafield)
                self.retrieved_ids.add(datafield_id)

        if new_results:
            all_results.extend(new_results)
            total_count += len(new_results)
            no_new_data_count = 0  # Reset the counter

            for index, datafield in enumerate(new_results):
                print(f"Datafield {index + 1}: {datafield}")
        else:
            no_new_data_count += 1
            print("No new datafields found in this batch.")

        if total_count >= max_volume:
            print(f"Reached maximum volume of {max_volume} datafields.")
            break

        # Break if no new data has been found for a certain number of iterations
        if no_new_data_count >= 3:  # Adjust this threshold as needed
            print("No new datafields found for several iterations. Exiting loop.")
            break

        print(f'Total datafields retrieved so far: {total_count}')
        offset += batch_size  # Increment offset for the next batch

    self.save_retrieved_ids()
    print(f'Total datafields retrieved: {total_count}')
    return all_results

BRAINAPIWRAPPER.retrieve_datafields3 = retrieve_datafields3


# In[7]:


def get_datafields(
    self,
    instrument_type: str = 'EQUITY',
    region: str = 'GLB',
    delay: int = 1,
    universe: str = 'MINVOL1M',
    dataset_id: str = '',
    search: str = '',
    batch_size: int = None
#   limit: int = 20  # Add a limit parameter with a default value
):
    if len(search) == 0:
        url_template = brain_api_url + "/data-fields?" +\
            f"&instrumentType={instrument_type}" +\
            f"&region={region}&delay={str(delay)}&universe={universe}&dataset.id={dataset_id}&limit=50" +\
            "&offset={x}"
        count = self.session.get(url_template.format(x=0)).json()['count'] 
    else:
        url_template = brain_api_url + "/data-fields?" +\
            f"&instrumentType={instrument_type}" +\
            f"&region={region}&delay={str(delay)}&universe={universe}&limit=50" +\
            f"&search={search}" +\
            "&offset={x}"
        count = 100
    
    max_try=5
    datafields_list = []
    for x in range(0, count, 50):
        for _ in range(max_try):
            datafields = self.session.get(url_template.format(x=x))
            if 'results' in datafields.json():
                break
            else:
                time.sleep(5)
            
        datafields_list.append(datafields.json()['results'])

    datafields_list_flat = [item for sublist in datafields_list for item in sublist]

    datafields_df = pd.DataFrame(datafields_list_flat)
    print(datafields_df)    
    return datafields_df


# Add the new method to the BRAINAPIWRAPPER class
BRAINAPIWRAPPER.get_datafields = get_datafields


# In[8]:


def get_and_save_datafields(
    self,
    instrument_type: str = 'EQUITY',
    region: str = 'GLB',
    delay: int = 1,
    universe: str = 'MINVOL1M',
    dataset_id: str = '',
    search: str = '',
    batch_size: int = None
):
    if len(search) == 0:
        url_template = brain_api_url + "/data-fields?" +\
            f"&instrumentType={instrument_type}" +\
            f"&region={region}&delay={str(delay)}&universe={universe}&dataset.id={dataset_id}&limit=50" +\
            "&offset={x}"
        count = self.session.get(url_template.format(x=0)).json()['count'] 
    else:
        url_template = brain_api_url + "/data-fields?" +\
            f"&instrumentType={instrument_type}" +\
            f"&region={region}&delay={str(delay)}&universe={universe}&limit=50" +\
            f"&search={search}" +\
            "&offset={x}"
        count = 100
    
    max_try = 5
    datafields_list = []
    for x in range(0, count, 50):
        for _ in range(max_try):
            datafields = self.session.get(url_template.format(x=x))
            if 'results' in datafields.json():
                break
            else:
                time.sleep(5)
            
        datafields_list.append(datafields.json()['results'])

    datafields_list_flat = [item for sublist in datafields_list for item in sublist]

    datafields_df = pd.DataFrame(datafields_list_flat)
    print(datafields_df)    

    # Generate a filename based on parameters
    filename = f"datafields_{instrument_type}_{region}_{universe}.csv"
    
    # Save the DataFrame to a CSV file
    datafields_df.to_csv(filename, mode='w', header=True, index=False)  # Write with header
    print(f"Data fields saved to {filename}")

    return datafields_df

# Add the new method to the BRAINAPIWRAPPER class
BRAINAPIWRAPPER.get_and_save_datafields = get_and_save_datafields


# In[6]:


def save_datafields_to_csv_file(datafields):
    # Convert results to DataFrame
    datafields_df = pd.DataFrame(datafields)

    # Write the DataFrame to a CSV file, I REMOVED OUTPUT FILE
    datafields_df.to_csv(output_file, mode='w', header=True, index=False)  # Write with header

BRAINAPIWRAPPER.save_datafields_to_csv_file = save_datafields_to_csv_file


# In[9]:


all_datafields=s.get_and_save_datafields()
print(f"Number of data fields retrieved: {len(all_datafields)}")
print(f"s: {s}")  # Check if s is defined correctly

#if all_datafields:
#    s.save_datafields_to_csv_file(datafields, output_file='datafieldsJPN.csv')


# In[10]:


datafields = s.retrieve_datafields3(batch_size=50)
#datafields = s.retrieve_datafields()

print(f"Number of data fields retrieved: {len(datafields)}")
print(f"s: {s}")  # Check if s is defined correctly

if datafields:
    s.save_datafields_to_csv_file(datafields, output_file='datafieldsGLBMINVOL.csv')


# Check if any data fields were retrieved
if datafields:
    # Print each retrieved data field
    for index, datafield in enumerate(datafields):
        print(f"Datafield {index + 1}: {datafield}")
else:
    print("No data fields retrieved.")



# In[ ]:


datafields = s.retrieve_datafields(batch_size=50)


# In[ ]:


#separate method for get vector datafield, matrix datafields, others...


# In[ ]:





# In[ ]:


def get_datasets(self, instrument_type: str = 'EQUITY', region: str = 'USA', delay: int = 1, universe: str = 'TOP3000', file_path: str = None):
    url = brain_api_url + "/data-sets?" +\
        f"instrumentType={instrument_type}&region={region}&delay={str(delay)}&universe={universe}"
    result = self.session.get(url)
    datasets_df = pd.DataFrame(result.json()['results'])
    
    # Save the DataFrame to a CSV file if a file path is provided
    if file_path:
        datasets_df.to_csv(file_path, index=False)
    
    return datasets_df

# Add the new method to the BRAINAPIWRAPPER class
BRAINAPIWRAPPER.get_datasets = get_datasets


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




