import re
import json
import glob
import gspread
import numpy as np
import logging as log
import tqdm as progress

from os import remove
from os.path import isfile
from datetime import datetime as date

from process import process_features
from process import write_log

class DataRetriever():
    def __init__(self, verbose):
        self.verbose = verbose
        # Read config file
        with open("config.json") as d:
            self.data = json.load(d)
            self.nations = self.data["nations"]
            self.sheet_id = self.data["sheet"]
            self.force_reset = self.data["force_data_reset"]
            self.prefix = self.data["data_prefix"]
            self.model_prefix = self.data["model_prefix"]
            self.suffix = {}
            self.suffix['l'] = self.data["label_suffix"]
            self.suffix['d'] = self.data["date_suffix"]
            self.suffix['1'] = self.data["ore1_suffix"]
            self.suffix['2'] = self.data["ore2_suffix"]
            self.suffix['f'] = self.data["feature_suffix"]
            self.suffix['y'] = self.data["ylabel_suffix"]
            self.suffix['b'] = self.data["blabel_suffix"]
            
        # Authenticate Google Sheets API
        write_log("Data: Connecting to Sheets...", log.info, True, verbose, end="\r")
        if not isfile("key.json"):
            write_log("File \"key.json\" does not exist. Please create the file with your Google Sheets credentials.", log.error, False, False)
            exit(1)
        self.client = gspread.service_account("key.json")
        try:
            self.ws = self.client.open_by_key(self.sheet_id)
        except:
            write_log("The sheet ID in the config is not valid with your account.", log.error, False, False)
            exit(1)
        
        # Read data from files
        self.read_data()
    
    def read_data(self):
        list_order = ['d', '1', '2', 'f', 'y', 'b']
        
        self.ex_data = []
        if not self.force_reset:
            for n in self.nations:
                write_log(f"Data: Reading { n } data...", log.info, True, self.verbose, end="\r")
                nation_data = {}
                for k in list_order:
                    filename = self.prefix + n.lower() + self.suffix[k] + ".npy"
                    nation_data[k] = np.array(np.load(filename)).tolist() if isfile(filename) else []
                self.ex_data.append(nation_data)
            write_log("Data: Reading nation data complete.", log.info, True, self.verbose)
        else:
            write_log("Data: Force reset initiated...", log.warning, False, self.verbose)
            self.ex_data = [{ k : [] for k in list_order } for _ in self.nations]
            
            files = glob.glob(self.prefix + "*")
            for f in files:
                remove(f)

            with open("config.json", 'w') as config:
                self.data["force_data_reset"] = False
                json.dump(self.data, config, indent=2)
    
    def retrieve_mining_data(self):
        write_log("Data: Constructing nation data...", log.info, True, self.verbose)
        for i, n in enumerate(self.nations):
                
            # Open the Google Sheets document
            sheet = self.ws.worksheet(n + " Data")
            
            # Find column for leyline screenshots
            leyline_col = sheet.find("Leyline", 1).col
            # Get all image URLs
            data_amount = len(sheet.col_values(leyline_col)[2:])
            # Check if current data is sufficent
            current_amount = len(self.ex_data[i]["d"])
            if data_amount <= current_amount:
                write_log(f"Data: {n} looks done already.", log.info, True, self.verbose, end='\r')
                continue
            write_log(f"Data: Working on {n}...", log.info, True, self.verbose, end='\r')
            
            # Get all date data
            dates = sheet.col_values(1)[2 + current_amount:2 + data_amount]
            dates = [date.strptime(d, '%m/%d/%y') for d in dates]
            self.ex_data[i]["d"].extend([[d.year, d.month, d.day] for d in dates])
            
            # Get all ore data
            ore_data = sheet.range(3 + current_amount, 2, 3 + data_amount, leyline_col - 1)
            # Split range data by row
            ores = [ore_data[i:i + leyline_col - 2] for i in range(0, len(ore_data), leyline_col - 2)]
            # Get cell values instead of Cell objects
            ores = [[(int(x.value) if re.search('[0-9]', x.value) else '') for x in y] for y in ores]
            # Trim to only indices, remove blanks
            ores = [[[i for i, y in enumerate(x) if y == 1], [i for i, y in enumerate(x) if y == 2]] for x in ores if 1 in x]
            self.ex_data[i]['1'].extend([x[0] for x in ores])
            self.ex_data[i]['2'].extend([x[1] for x in ores])
            
            # Get all image URLs
            image_urls = sheet.col_values(leyline_col)[2 + current_amount: 2 + data_amount]
            
            # Find column for end of sheet data
            end_col = sheet.find("2", 1).col
            # Get all labels [[y,b], [y,b], ...]
            label_data = sheet.range(3 + current_amount, leyline_col + 1, 3 + data_amount, end_col - 1)
            # Split range data by row
            leylines = [label_data[i:i + end_col - 1 - leyline_col] for i in range(0, len(label_data), end_col - 1 - leyline_col)]
            # Get cell values instead of Cell objects
            leylines = [[x.value for x in y] for y in leylines]
            # Trim to only indices, remove blanks
            leylines = [[x.index('y'), x.index('b')] for x in leylines if 'y' in x]
            self.ex_data[i]['y'].extend([x[0] for x in leylines])
            self.ex_data[i]['b'].extend([x[1] for x in leylines])

            write_log(f"Data: Writing info for { n }", log.info, True, self.verbose, end="\r")

            # Order to store features and labels
            list_order = ['d', '1', '2', 'y', 'b']
            # Save features and labels to .npy files
            for l in list_order:
                np.save(self.prefix + n.lower() + self.suffix[l] + ".npy", np.array(self.ex_data[i][l]))

            # Iterate through each image URL
            for url in progress.tqdm(image_urls, desc=n + " Image Progress", total=len(image_urls)):
                # Get image features from url
                self.ex_data[i]['f'].append(process_features(url))
            # Save features to .npy file
            np.save(self.prefix + n.lower() + self.suffix['f'] + ".npy", np.array(self.ex_data[i]['f']))
                
            write_log(f"Data: { n } completed.", log.info, True, self.verbose)
            
        write_log("Data: Features and labels saved successfully.", log.info, False, self.verbose)

        # Write the timestamp for when data was updated
        data = {}
        with open(self.model_prefix + "config.json", "r") as config:
            data = json.load(config)
        data["Data"] = date.now().timestamp()
        with open(self.model_prefix + "config.json", "w") as config:
            json.dump(data, config, indent=2)
            
        write_log("Data: Timestamp updated.", log.info, False, self.verbose)

# Main function
def main():
    log.basicConfig(format="%(message)s", level=log.INFO)
    data = DataRetriever(True)
    data.retrieve_mining_data()

# Call the main function
if __name__ == "__main__":
    main()
