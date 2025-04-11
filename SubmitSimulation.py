#!/usr/bin/env python
# coding: utf-8

# In[ ]:


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
import random
import logging
import time
import random
import csv

# Configure logging to output to both console and a log file
log_filename = "simulation_log.txt"
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(message)s",
    handlers=[
        logging.FileHandler(log_filename, mode='w'),  # Write logs to a file
        logging.StreamHandler()  # Also print logs to the console
    ]
)

brain_api_url = os.environ.get("BRAIN_API_URL", "https://api.worldquantbrain.com")
brain_url = os.environ.get("BRAIN_URL", "https://platform.worldquantbrain.com") 

class RateLimitExceededError(Exception):
    """Custom exception for rate limit errors."""
    pass

class BRAINAPIWRAPPER:
    
    def __init__(self):
        self.session = self.get_login_session()
        self.permissions = self.check_permissions()  # Initialize permissions here

        
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
    


# In[ ]:


s = BRAINAPIWRAPPER()


# In[ ]:


def get_data_type(self, item):
    # Check item metadata or structure to identify type
    # This is an example; adapt based on your data structure.
    if hasattr(item, 'data_type'):
        return item.data_type  # Assume each item has a 'data_type' attribute
    # Or use other logic as needed
    return 'unknown'

BRAINAPIWRAPPER.get_data_type = get_data_type


# In[ ]:


def load_first_column_from_csv(filename):
    """
    Reads a CSV file and returns the first column as a DataFrame.

    Parameters:
    filename (str): The path to the CSV file.

    Returns:
    DataFrame: A DataFrame containing the first column of the CSV file.
    """
    try:
        print(f"Type of filename: {type(filename)}")  # This should print <class 'str'>

        # Read the CSV file
        df = pd.read_csv(filename)

        # Load the first column into a new DataFrame
        first_column_df = df.iloc[:, [0]]  # iloc[:, [0]] selects the first column

        return first_column_df
    except Exception as e:
        print(f"An error occurred: {e}")
        return None

BRAINAPIWRAPPER.load_first_column_from_csv = load_first_column_from_csv


# In[ ]:


def load_first_column_with_type_check(filename):
    """
    Reads a CSV file and returns the first column, filtering only 'vector' entries if they exist.

    Parameters:
    filename (str): The path to the CSV file.

    Returns:
    DataFrame: A DataFrame containing only rows where the first column type is 'vector'.
    """
    try:
        # Read the entire CSV file into a DataFrame
        df = pd.read_csv(filename)
        
        # Assume the second column contains type information (e.g., 'vector' or 'matrix')
        if 'type' in df.columns:
            # Filter for rows where 'type' is 'vector'
            filtered_df = df[df['type'] == 'matrix'].iloc[:, [0]]
        else:
            print("No 'type' column found. Returning first column without filtering.")
            filtered_df = df.iloc[:, [0]]  # No filtering, only first column

        return filtered_df
    except Exception as e:
        print(f"An error occurred: {e}")
        return None

BRAINAPIWRAPPER.load_first_column_with_type_check = load_first_column_with_type_check


# In[ ]:


def load_and_print_first_column_with_matrix_type(filename):
    """
    Reads a CSV file, prints each row's type, and returns only rows where type is 'matrix'.

    Parameters:
    filename (str): The path to the CSV file.

    Returns:
    DataFrame: A DataFrame containing the first column for rows where type is 'matrix'.
    """
    try:
        # Read the entire CSV file into a DataFrame
        df = pd.read_csv(filename)

        # Check if there's a 'type' column
        if 'type' in df.columns:
            # Initialize an empty list to store rows with 'matrix' type
            matrix_rows = []

            # Iterate over each row and print the type
            for index, row in df.iterrows():
                row_type = row['type']  # Get the type from the 'type' column
                first_column_value = row.iloc[0]  # Get the value in the first column

                # Print the row type and first column value for inspection, comment out as too long
                # print(f"Row {index}: Type = {row_type}, First Column Value = {first_column_value}")

                # Only add rows with 'matrix' type to the list
                if row_type == 'MATRIX':
                    matrix_rows.append(row)

            # Create a DataFrame from the filtered rows
            filtered_df = pd.DataFrame(matrix_rows)
            # Return only the first column from the filtered DataFrame
            return filtered_df.iloc[:, [0]]
        
        else:
            print("No 'type' column found. Returning the first column without filtering.")
            # No filtering; return the first column as-is
            return df.iloc[:, [0]]

    except Exception as e:
        print(f"An error occurred: {e}")
        return None

BRAINAPIWRAPPER.load_and_print_first_column_with_matrix_type = load_and_print_first_column_with_matrix_type


# In[ ]:


def load_and_print_first_column_with_matrix_type2(filename):

    try:
        # Read the CSV file into a DataFrame
        df = pd.read_csv(filename)
        print("CSV File Loaded Successfully. Preview:")
        print(df.head())

        # Ensure the 'type' column exists (case insensitive check)
        if 'type' not in df.columns:
            print("No 'type' column found. Returning the first column without filtering.")
            return df.iloc[:, [0]]

        # Debug: Print the entire 'type' column
        print("Type column values:")
        print(df['type'])

        # Initialize an empty list to store rows with 'MATRIX' type
        matrix_rows = []

        # Iterate over each row and print its details
        for index, row in df.iterrows():
            # Debug: Inspect the row
            print(f"Debugging Row {index}: {row}")

            # Check if 'type' is valid
            if pd.isna(row['type']) or not isinstance(row['type'], str):
                print(f"Skipping Row {index}: Invalid 'type' value = {row['type']}")
                continue

            # Normalize the 'type' column value for comparison
            row_type = str(row['type']).strip().upper()
            first_column_value = row.iloc[0]

            # Debug: Print details of the current row , commented out as too long
            # print(f"Row {index}: Type = {row_type}, First Column Value = {first_column_value}")

            # Add rows with 'MATRIX' type to the list
            if row_type == 'MATRIX':
                matrix_rows.append(row)

        # Create a DataFrame from the filtered rows
        filtered_df = pd.DataFrame(matrix_rows)

        # Return only the first column from the filtered DataFrame
        return filtered_df.iloc[:, [0]]

    except Exception as e:
        print(f"An error occurred: {e}")
        return None

BRAINAPIWRAPPER.load_and_print_first_column_with_matrix_type2 = load_and_print_first_column_with_matrix_type2


# In[ ]:


def load_and_print_first_column_with_vector_type(filename):
    """
    Reads a CSV file, prints each row's type, and returns only rows where type is 'vector'.

    Parameters:
    filename (str): The path to the CSV file.

    Returns:
    DataFrame: A DataFrame containing the first column for rows where type is 'matrix'.
    """
    try:
        # Read the entire CSV file into a DataFrame
        df = pd.read_csv(filename)

        # Check if there's a 'type' column
        if 'type' in df.columns:
            # Initialize an empty list to store rows with 'matrix' type
            vector_rows = []

            # Iterate over each row and print the type
            for index, row in df.iterrows():
                row_type = row['type']  # Get the type from the 'type' column
                first_column_value = row.iloc[0]  # Get the value in the first column

                # Print the row type and first column value for inspection, comment out as too long
                # print(f"Row {index}: Type = {row_type}, First Column Value = {first_column_value}")

                # Only add rows with 'matrix' type to the list
                if row_type == 'VECTOR':
                    matrix_rows.append(row)

            # Create a DataFrame from the filtered rows
            filtered_df = pd.DataFrame(vector_rows)
            # Return only the first column from the filtered DataFrame
            return filtered_df.iloc[:, [0]]
        
        else:
            print("No 'type' column found. Returning the first column without filtering.")
            # No filtering; return the first column as-is
            return df.iloc[:, [0]]

    except Exception as e:
        print(f"An error occurred: {e}")
        return None

BRAINAPIWRAPPER.load_and_print_first_column_with_vector_type = load_and_print_first_column_with_vector_type


# In[ ]:


#REQUIRED
def create_simulation_data(self, first_column):
    """
    Creates a simulation data dictionary with the first column.

    Parameters:
    first_column (list): The first column data.

    Returns:
    dict: A simulation data dictionary.
    """
    # Convert the first column list to a string format for the 'regular' field
    first_column_str = ', '.join(map(str, first_column))  # Convert each item to string and join

    simulation_data = {
        'type': 'REGULAR',
        'settings': {
            'instrumentType': 'EQUITY',
            'region':'ASI',
            'universe': 'MINVOL1M',
            'delay': 1,
            'decay' : 4,
            'neutralization': 'NONE',
            'truncation': 0.08,
            'pasteurization': 'ON',
            'testPeriod': 'P6Y0M0D',
            'unitHandling': 'VERIFY',
            'nanHandling': 'OFF',
            'language': 'FASTEXPR',
            'visualization': False,
        },
        'regular': f'ts_rank(({first_column_str}), 20)'  # Format the string
        #'regular': f'ts_decay_linear({first_column_str}, 10) * rank(volume*close) + ts_decay_linear({first_column_str}, 50) * (1 - rank(volume*close))'
    }

    return simulation_data


BRAINAPIWRAPPER.create_simulation_data = create_simulation_data


# In[ ]:


#REQUIRED
def send_simulation(self, simulation_settings):

    # Check if simulation_settings is a dictionary
    if not isinstance(simulation_settings, dict):
        raise ValueError("simulation_settings must be a dictionary.")
    
    simulation_response = self.session.post ('https://api.worldquantbrain.com/simulations',json=simulation_settings)
#    simulation_response = self.session.post ('https://api.worldquantbrain.com/simulations',json=truncated_alpha_list2)

    #Comment out as printout too long
    #print(simulation_response.headers)
    print(simulation_response.text)
    
    location = simulation_response.headers.get('Location')
    if location is not None:
        print("Location: " + location)
    else:
        print("Warning: 'Location' header not found in the response. Continuing to the next simulation.")
        return None  # Return None or handle as needed, but continue processing
    simulation_location = simulation_response.headers['Location']

    return simulation_location

#original code simulation_progress_url changed to simulation_location
BRAINAPIWRAPPER.send_simulation = send_simulation


# In[ ]:


simulation_location_response = s.send_simulation(simulation_data)
print("Location Response: " + simulation_location_response )


# In[ ]:


#OLD
def check_progress_return_alpha(self, simulation_location_response):

    while True:
        simulation_progress = self.session.get(simulation_location_response)

        # comment out as printout too long
        # print("Simulation Progress Response:", simulation_progress.json())
        
        if simulation_progress.headers.get("Retry-After",0) == 0:
            break
        #Commented as printout is too long
        #print ("Sleeping for " + simulation_progress.headers["Retry-After"] + " secs.")
        sleep(float(simulation_progress.headers["Retry-After"]))
    json_data = simulation_progress.json()
#    print("Simulation progress JSON:", json_data)        # HERE

    try:
        alpha_id = json_data["alpha"]
    except KeyError:
        print("Error: 'alpha' key not found in the response.")
        return None  # or handle it in another way, e.g., raise an exception

BRAINAPIWRAPPER.check_progress_return_alpha = check_progress_return_alpha


# In[ ]:


#REVISED 20241027
#REQUIRED
def check_progress_return_alpha(self, simulation_location_response):

    while True:
        simulation_progress = self.session.get(simulation_location_response)
        json_data = simulation_progress.json()
        
        # comment out as printout too long
        # print("Simulation Progress Response:", json_data)

        # Check if the simulation is complete
        if json_data.get("status") == "COMPLETE":
            try:
                alpha_id = json_data["alpha"]
                print(f"Check Progress Returned alpha_id: {alpha_id}")  # Debug: Confirm alpha_id retrieved
                return alpha_id  # Return the alpha_id if the simulation is complete
            except KeyError:
                print("ERROR: 'alpha' key not found in the response.")
                return None

        # Handle Retry-After for ongoing simulation
        retry_after = simulation_progress.headers.get("Retry-After")
        if retry_after:
            # Commented out as printout is too long
            # print(f"Sleeping for {retry_after} secs.")
            sleep(float(retry_after))
        else:
            print("No Retry-After header found; waiting 2 seconds by default.")
            sleep(2)  # Default delay if no "Retry-After" header is provided

BRAINAPIWRAPPER.check_progress_return_alpha = check_progress_return_alpha


# In[ ]:


alpha_response = s.check_progress_return_alpha(simulation_location_response)


# In[ ]:


#REQUIRED

def simulate_alpha(self, simulation_settings):
    simulation_location = self.send_simulation(simulation_settings)
    alpha_id = self.check_progress_return_alpha(simulation_location)
    simulation_results = self.get_results_dictionary(alpha_id)
    return simulation_results

BRAINAPIWRAPPER.simulate_alpha = simulate_alpha


# In[ ]:


#REQUIRED

def get_results_dictionary(self, alpha_id):
# Check if alpha_id is valid
    if not alpha_id:  # This checks for None, empty string, or any falsy value
        print("Get results dict Error: Invalid alpha_id. Please provide a valid alpha_id.")
        return None  # Or raise an exception, depending on your error handling strategy

    simulation_results = {
        'summary_results aka get_results_dictionary': self.get_results(alpha_id),
        'submission_checks aka get_results_dictionary': self.get_results(alpha_id, stats="/check")
    }

    # commenting as logging too long
    # print(simulation_results)

    #print to file
    logging.info(f"simu_results: {simulation_results}\n\n")
    
    return simulation_results

BRAINAPIWRAPPER.get_results_dictionary = get_results_dictionary


# In[ ]:


def save_results(self, simulation_results):

    # Convert the results to a JSON string
    dict_string2 = json.dumps(simulation_results, ensure_ascii=False)
    
    # Print the modified string for debugging
    print("dict_string: " + dict_string2)
    
    current_date = datetime.now().strftime('%Y-%m-%d')
    filename = f'datasets_{current_date}.txt'
    
    # Open the file in append mode
    with open(filename, 'a') as file:
        file.write(dict_string2 + "\n\n")  # Append the results and add a newline for separation

    return simulation_results

BRAINAPIWRAPPER.save_results = save_results


# In[ ]:


#REQUIRED

def get_results(self, alpha_id, stats=""):
    results = self.session.get("https://api.worldquantbrain.com/alphas/" + alpha_id + stats)
#    print("results: " + results)
    print("\n")

    
    print("get_results: " + results.text, "\n\n")  # Use .text or .json() to print the response content    

    #print to file
    logging.info(f"get_results: {results.text}\n\n")

    print("\n")
    return results.json()
BRAINAPIWRAPPER.get_results = get_results


# In[ ]:


def get_results(self, alpha_id, stats=""):
    try:
        # Make the API request
        results = self.session.get(f"https://api.worldquantbrain.com/alphas/{alpha_id}{stats}")

        # Check for rate limit error in the response
        if "No Retry-After header found" in results.text or results.status_code == 429:
            logging.error("Rate limit exceeded. Stopping further processing.")
            raise RateLimitExceededError("Rate limit exceeded. API requests halted.")

        # Print and log the results
        # print(f"get_results: {results.text}\n\n")
        logging.info(f"get_results: {results.text}\n\n")

        # Return the JSON response
        return results.json()
    
    except RateLimitExceededError:
        raise  # Re-raise the custom exception for higher-level handling
    except Exception as e:
        logging.error(f"An error occurred in get_results: {e}")

BRAINAPIWRAPPER.get_results = get_results


# In[ ]:


# Define the main batch processing function
def process_simulations_in_batches(filename, batch_size=3):

    # Print type to confirm
    print(f"In process_simulations_in_batches, type of filename: {type(filename)}")

    # Verify filename is a string
    if not isinstance(filename, str):
        print("Error: filename must be a string representing the file path.")
        return
    
    # Step 1: Load the first column of data from the CSV file
    # first_column_df = load_first_column_from_csv(filename)
    first_column_df = load_first_column_with_type_check(filename)
    if first_column_df is None:
        print("Failed to load data from CSV.")
        return

    first_column_list = first_column_df.iloc[:, 0].tolist()  # Convert to list

    # Step 2: Process in batches
    results = []
    for i in range(0, len(first_column_list), batch_size):
        batch = first_column_list[i:i + batch_size]
        
        for alpha in batch:
            # Step 3: Create simulation data for each alpha
            print("current expression: " + alpha)        
            simulation_settings = s.create_simulation_data([alpha])
            
            # Step 4: Send the simulation and check for progress
            simulation_location = s.send_simulation(simulation_settings)
            if simulation_location is None:
                print("Simulation initiation failed, moving to the next item.")
                continue  # Skip to the next item in the batch if initiation failed
            
            # Step 5: Monitor progress and retrieve alpha ID
            alpha_id = s.check_progress_return_alpha(simulation_location)
            if alpha_id is None:
                print(f"Failed to retrieve alpha_id for {alpha}.")
                continue
            
            # Step 6: Get results for the completed simulation
            simulation_result = s.get_results_dictionary(alpha_id)
            results.append({alpha: simulation_result})
            print(f"Number of alphas returned so far: {len(results)}")
            
            # Optional: small delay between individual simulations
            time.sleep(random.uniform(0.5, 1.5))
        
        print(f"Processed batch {(i // batch_size) + 1} of {len(first_column_list) // batch_size + 1}")

        # Step 7: Delay between batches to manage API load
        time.sleep(random.uniform(1, 2))  # Adjust as necessary for API rate limits

    return results

BRAINAPIWRAPPER.process_simulations_in_batches = process_simulations_in_batches


# In[ ]:


# Define the main batch processing function
def process_simulations_in_batches2(filename, batch_size=3):
    print(f"In process_simulations_in_batches, type of filename: {type(filename)}")

    if not isinstance(filename, str):
        print("Error: filename must be a string representing the file path.")
        return
    
    # first_column_df = load_first_column_from_csv(filename)
    first_column_df = load_and_print_first_column_with_matrix_type(filename)
#    first_column_df = load_and_print_first_column_with_vector_type(filename)
    if first_column_df is None:
        print("Failed to load data from CSV.")
        return

    first_column_list = first_column_df.iloc[:, 0].tolist()

    # Store all results that meet the criteria
    all_results = []
    for i in range(0, len(first_column_list), batch_size):
        batch = first_column_list[i:i + batch_size]
        
        current_batch_results = []

        for alpha in batch:
            print("Current expression: " + alpha)        
            simulation_settings = s.create_simulation_data([alpha])
            
            simulation_location = s.send_simulation(simulation_settings)
            if simulation_location is None:
                print("Simulation initiation failed, moving to the next item.")
                continue
            
            alpha_id = s.check_progress_return_alpha(simulation_location)
            if alpha_id is None:
                print(f"Failed to retrieve alpha_id for {alpha}.")
                continue
            
            simulation_result = s.get_results_dictionary(alpha_id)

            sharpe_ratio = simulation_result.get("sharpe_ratio", None)
            if sharpe_ratio is not None and sharpe_ratio > 0.7:
                current_batch_results.append({
                    "alpha_id": alpha_id,
                    "expression": alpha,
                    "sharpe_ratio": sharpe_ratio,
                    "universe": simulation_result.get("universe"),
                    "region": simulation_result.get("region"),
                    "turnover": simulation_result.get("turnover"),
                    "drawdown": simulation_result.get("drawdown"),
                    "fitness": simulation_result.get("fitness"),
                })
                # Append valid results to all_results
                all_results.append({
                    "alpha_id": alpha_id,
                    "expression": alpha,
                    "sharpe_ratio": sharpe_ratio,
                    "universe": simulation_result.get("universe"),
                    "region": simulation_result.get("region"),
                    "turnover": simulation_result.get("turnover"),
                    "drawdown": simulation_result.get("drawdown"),
                    "fitness": simulation_result.get("fitness"),
                })
                print(f"Sharpe Ratio {sharpe_ratio} is greater than 1.1 for alpha: {alpha}")

            time.sleep(random.uniform(0.5, 1.5))
        
        if current_batch_results:
            output_filename = f"batch_results_{i // batch_size + 1}.csv"  
            with open(output_filename, mode='w', newline='') as file:
                writer = csv.DictWriter(file, fieldnames=current_batch_results[0].keys())
                writer.writeheader()
                writer.writerows(current_batch_results)

            print(f"Batch results written to {output_filename}")
        else:
            print(f"No results to write for batch {i // batch_size + 1}.")

        print(f"Processed batch {(i // batch_size) + 1} of {len(first_column_list) // batch_size + 1}")

        time.sleep(random.uniform(1, 2))

    # Save all_results with the date in the filename
    if all_results:
        today_date = datetime.now().strftime("%Y%m%d")  # Get today's date in YYYYMMDD format
        all_results_filename = f"{today_date}.csv"  # Create the filename with today's date
        with open(all_results_filename, mode='w', newline='') as file:
            writer = csv.DictWriter(file, fieldnames=all_results[0].keys())
            writer.writeheader()
            writer.writerows(all_results)
        print(f"All valid results saved to {all_results_filename}")

BRAINAPIWRAPPER.process_simulations_in_batches2 = process_simulations_in_batches2


# In[ ]:


def process_simulations_in_batches3(filename, batch_size=3):
    logging.info(f"In process_simulations_in_batches, type of filename: {type(filename)}")

    if not isinstance(filename, str):
        logging.error("Error: filename must be a string representing the file path.")
        return
    
    # first_column_df = load_first_column_from_csv(filename)
    first_column_df = load_and_print_first_column_with_matrix_type(filename)
    #    first_column_df = load_and_print_first_column_with_vector_type(filename)
    if first_column_df is None:
        logging.error("Failed to load data from CSV.")
        return

    first_column_list = first_column_df.iloc[:, 0].tolist()

    # Store all results that meet the criteria
    all_results = []
    for i in range(0, len(first_column_list), batch_size):
        batch = first_column_list[i:i + batch_size]
        
        current_batch_results = []

        for alpha in batch:
            logging.info(f"Current expression: {alpha}")        
            simulation_settings = s.create_simulation_data([alpha])
            
            simulation_location = s.send_simulation(simulation_settings)
            if simulation_location is None:
                logging.warning("Simulation initiation failed, moving to the next item.")
                continue
            
            alpha_id = s.check_progress_return_alpha(simulation_location)
            if alpha_id is None:
                logging.warning(f"Failed to retrieve alpha_id for {alpha}.")
                continue
            
            simulation_result = s.get_results_dictionary(alpha_id)

            sharpe_ratio = simulation_result.get("sharpe_ratio", None)
            if sharpe_ratio is not None and sharpe_ratio > 0.7:
                current_batch_results.append({
                    "alpha_id": alpha_id,
                    "expression": alpha,
                    "sharpe_ratio": sharpe_ratio,
                    "universe": simulation_result.get("universe"),
                    "region": simulation_result.get("region"),
                    "turnover": simulation_result.get("turnover"),
                    "drawdown": simulation_result.get("drawdown"),
                    "fitness": simulation_result.get("fitness"),
                })
                # Append valid results to all_results
                all_results.append({
                    "alpha_id": alpha_id,
                    "expression": alpha,
                    "sharpe_ratio": sharpe_ratio,
                    "universe": simulation_result.get("universe"),
                    "region": simulation_result.get("region"),
                    "turnover": simulation_result.get("turnover"),
                    "drawdown": simulation_result.get("drawdown"),
                    "fitness": simulation_result.get("fitness"),
                })
                logging.info(f"Sharpe Ratio {sharpe_ratio} is greater than 0.7 for alpha: {alpha}")

            time.sleep(random.uniform(0.5, 1.5))
        
        if current_batch_results:
            output_filename = f"batch_results_{i // batch_size + 1}.csv"  
            with open(output_filename, mode='w', newline='') as file:
                writer = csv.DictWriter(file, fieldnames=current_batch_results[0].keys())
                writer.writeheader()
                writer.writerows(current_batch_results)

            logging.info(f"Batch results written to {output_filename}")
        else:
            logging.info(f"No results to write for batch {i // batch_size + 1}.")

        logging.info(f"Processed batch {(i // batch_size) + 1} of {len(first_column_list) // batch_size + 1}")

        # Save logs before hitting rate limits
        logging.info(f"Pausing to handle rate limit.")
        time.sleep(random.uniform(1, 2))

    # Save all_results with the date in the filename
    if all_results:
        today_date = datetime.now().strftime("%Y%m%d")  # Get today's date in YYYYMMDD format
        all_results_filename = f"{today_date}_all_results.csv"  # Create the filename with today's date
        with open(all_results_filename, mode='w', newline='') as file:
            writer = csv.DictWriter(file, fieldnames=all_results[0].keys())
            writer.writeheader()
            writer.writerows(all_results)
        logging.info(f"All valid results saved to {all_results_filename}")

BRAINAPIWRAPPER.process_simulations_in_batches3 = process_simulations_in_batches3


# In[ ]:


def process_simulations_in_batches4(filename, batch_size=3):
    logging.info(f"Processing simulations in batches. Filename type: {type(filename)}")

    if not isinstance(filename, str):
        logging.error("Error: filename must be a string representing the file path.")
        return
    
    first_column_df = load_and_print_first_column_with_matrix_type(filename)
    if first_column_df is None:
        logging.error("Failed to load data from CSV.")
        return

    first_column_list = first_column_df.iloc[:, 0].tolist()

    all_results = []
    for i in range(0, len(first_column_list), batch_size):
        batch = first_column_list[i:i + batch_size]
        logging.info(f"Processing batch {i // batch_size + 1} with {len(batch)} items.")

        current_batch_results = []
        for alpha in batch:
            logging.info(f"Processing expression: {alpha}")
            try:
                simulation_settings = s.create_simulation_data([alpha])
                simulation_location = s.send_simulation(simulation_settings)

                if simulation_location is None:
                    logging.warning("Simulation initiation failed, skipping this expression.")
                    continue
                
                alpha_id = s.check_progress_return_alpha(simulation_location)
                if alpha_id is None:
                    logging.warning(f"Failed to retrieve alpha_id for expression: {alpha}.")
                    continue

                simulation_result = s.get_results_dictionary(alpha_id)

                sharpe_ratio = simulation_result.get("sharpe_ratio", None)
                if sharpe_ratio is not None and sharpe_ratio > 0.7:
                    result = {
                        "alpha_id": alpha_id,
                        "expression": alpha,
                        "sharpe_ratio": sharpe_ratio,
                        "universe": simulation_result.get("universe"),
                        "region": simulation_result.get("region"),
                        "turnover": simulation_result.get("turnover"),
                        "drawdown": simulation_result.get("drawdown"),
                        "fitness": simulation_result.get("fitness"),
                    }
                    current_batch_results.append(result)
                    all_results.append(result)
                    logging.info(f"Sharpe Ratio {sharpe_ratio} exceeds threshold for alpha: {alpha}")
                else:
                    logging.info(f"Sharpe Ratio {sharpe_ratio} is below threshold for alpha: {alpha}")

            except RateLimitExceededError as e:
                logging.error(f"Rate limit exceeded. Stopping processing: {e}")
                return
            except Exception as e:
                logging.error(f"An error occurred while processing {alpha}: {e}")
                continue

            time.sleep(random.uniform(0.5, 1.5))

        # Write batch results to a file
        if current_batch_results:
            output_filename = f"batch_results_{i // batch_size + 1}.csv"
            try:
                with open(output_filename, mode='w', newline='') as file:
                    writer = csv.DictWriter(file, fieldnames=current_batch_results[0].keys())
                    writer.writeheader()
                    writer.writerows(current_batch_results)
                logging.info(f"Batch results saved to {output_filename}")
            except Exception as e:
                logging.error(f"Failed to write batch results to {output_filename}: {e}")

        logging.info(f"Finished processing batch {i // batch_size + 1}.")
        time.sleep(random.uniform(1, 2))  # Pause to handle rate limits

    # Save all results to a consolidated file
    if all_results:
        all_results_filename = f"{datetime.now().strftime('%Y%m%d')}_all_results.csv"
        try:
            with open(all_results_filename, mode='w', newline='') as file:
                writer = csv.DictWriter(file, fieldnames=all_results[0].keys())
                writer.writeheader()
                writer.writerows(all_results)
            logging.info(f"All valid results saved to {all_results_filename}")
        except Exception as e:
            logging.error(f"Failed to save all results to {all_results_filename}: {e}")

BRAINAPIWRAPPER.process_simulations_in_batches4 = process_simulations_in_batches4


# In[ ]:


#RUN TO INPUT THE FILE AND RETRIEVE RESULTS AND SHARPE
filename = "datafields_EQUITY_ASI_MINVOL1M.csv"

# Call the function to process simulations in batches
process_simulations_in_batches4(filename, batch_size=3)  # Adjust batch_size as needed


# In[ ]:


# Prepare to store results
alpha_ids = []  # List to store alpha IDs
simulation_results_list = []  # List to store simulation results


# In[ ]:





# In[ ]:


# Define the duration for the simulation (5 minutes)
duration = 5 * 60  
end_time = time.time() + duration  # Calculate the end time


# In[ ]:


# Loop through each value in the first column, how does python for loop work? how to loop through each row in dataframe?
for index, row in .iterrows():
    first_row = row[0]  # Get the first column value
    simulation_data = create_simulation_data([first_row])  # Create simulation data for the current value
    print(f"Simulation data created for {first_row}:")
    print(simulation_data)

    # Send simulation and check progress
    simulation_location_response = s.send_simulation(simulation_data)
    print("Location Response: " + simulation_location_response)

    alpha_response = s.check_progress_return_alpha(simulation_location_response)
    print("Alpha Response: " + str(alpha_response))

    # Store the alpha ID in the list
    alpha_ids.append(alpha_response)

    # Simulate alpha and save results
    while time.time() < end_time:
        simulation_results = s.simulate_alpha(simulation_data)  # Assuming simulate_alpha is defined
        simulation_results_list.append(simulation_results)  # Store results
        s.save_results(simulation_results)  # Assuming save_results is defined
        time.sleep(15)  # Adjust the sleep time as needed

# Add the alpha IDs to the DataFrame
first_column_df['alpha_id'] = alpha_ids

# Save the updated DataFrame to a new CSV file
output_filename = "output_with_alpha_ids.csv"
first_column_df.to_csv(output_filename, index=False)
print(f"Alpha IDs saved to {output_filename}")


# In[ ]:


truncated_alpha_list2 =  [{'type': 'REGULAR', 'settings': {'instrumentType': 'EQUITY', 'region': 'USA', 'universe': 'TOP1000', 'delay': 1, 'decay': 20, 'neutralization': 'NONE', 'truncation': 0.08, 'pasteurization': 'ON', 'testPeriod': 'P1Y6M', 'unitHandling': 'VERIFY', 'nanHandling': 'OFF', 'language': 'FASTEXPR', 'visualization': False}, 'regular': 'vec_avg(anl16_medianest_normal)'}, {'type': 'REGULAR', 'settings': {'instrumentType': 'EQUITY', 'region': 'USA', 'universe': 'TOP1000', 'delay': 1, 'decay': 20, 'neutralization': 'NONE', 'truncation': 0.08, 'pasteurization': 'ON', 'testPeriod': 'P1Y6M', 'unitHandling': 'VERIFY', 'nanHandling': 'OFF', 'language': 'FASTEXPR', 'visualization': False}, 'regular': 'vec_avg(anl16_medianrec)'}]


# In[ ]:


json1 = json_output = json.dumps(truncated_alpha_list2, indent=4)
print (json1)


# In[ ]:


#simulate_alpha > send_simulation > check_progress_return_alpha > get_results_dictionary & save_results

while time.time() < end_time:
    for alpha1 in simulation_data2:
        print("ALL:" + str(alpha1))

#        simulation_settings=json1
        simulation_results = s.simulate_alpha(simulation_data2)
        s.save_results(simulation_results)
        time.sleep(15)  # Adjust the sleep time as needed


# In[ ]:


response1 = s.session.get("https://api.worldquantbrain.com/alphas/" + alpha_response + "/recordsets/pnl").json()
print(response1)
response2 = s.session.get("https://api.worldquantbrain.com/alphas/" + alpha_response + "/recordsets/sharpe").json()
print(response2)


# In[ ]:


def retrieve_datafields1(
    self,
    instrument_type: str = 'EQUITY',
    region: str = 'JPN',
    delay: int = 1,
    universe: str = 'TOP1600',
    dataset_id: str = '',
    search: str = '',
#    batch_size: int = 10  # Specify the batch size
    batch_size: int = 10
):
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
            print ("No results.")
            break  # Exit the loop if no more results

        # Highlighted Change: Print the retrieved results for this batch
        print(f"Retrieved batch of {len(results)} datafields:")  # New line added
        for index, datafield in enumerate(results):  # New loop added
            print(f"Datafield {index + 1}: {datafield}")  # New line added
        
        all_results.extend(results)  # Collect all results
        offset += batch_size
        total_count += len(results)

        # Optional: Print progress
        print(f'Retrieved {total_count} datafields so far...')

    print(f'Total datafields retrieved: {total_count}')
    return all_results  # Return all collected results

BRAINAPIWRAPPER.retrieve_datafields1 = retrieve_datafields1


# In[ ]:





# In[ ]:


datafields = s.retrieve_datafields1(batch_size=10)

# Check if any data fields were retrieved
if datafields:
    # Print each retrieved data field
    for index, datafield in enumerate(datafields):
        print(f"Datafield {index + 1}: {datafield}")
else:
    print("No data fields retrieved.")


# In[ ]:


#separate method for get vector datafield, matrix datafields, others...


# In[ ]:


def save_datafields_to_csv_file(datafields, output_file='datafields.csv'):
    # Convert results to DataFrame
    datafields_df = pd.DataFrame(datafields)

    # Write the DataFrame to a CSV file
    datafields_df.to_csv(output_file, mode='w', header=True, index=False)  # Write with header


# In[ ]:


def get_sharpe_ratio (self, alpha_response):

    self.session.get("https://api.worldquantbrain.com/alphas/" + alpha_response + "/recordsets/pnl").json()
    response = self.session.get("https://api.worldquantbrain.com/alphas/" + alpha_response + "/recordsets/pnl")
    print("Status Code:", response.status_code)
    if response.text:
        data = response.json()
    else:
        print("Error: Empty response body.")
    print("Response Headers:", response.headers)

    sharpe_ratio = None
    for check in alpha_checks['is']['checks']:
        if check['name'] == 'LOW_SHARPE':
            sharpe_ratio = check['value']
            break

    # Output the Sharpe ratio
    if sharpe_ratio is not None:
        print("Sharpe Ratio (from LOW_SHARPE check):", sharpe_ratio)
    else:
        print("Sharpe Ratio not found.")

    return sharpe_ratio

BRAINAPIWRAPPER.get_sharpe_ratio = get_sharpe_ratio


# In[ ]:


sharpe_ratio_response = s.get_sharpe_ratio(alpha_response)


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


Notes: check for matrix, ask chatgpt for example


# In[ ]:





# In[ ]:




