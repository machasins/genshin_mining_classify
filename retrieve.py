import re
import glob
import time
import numpy as np
import logging as log
import tqdm as progress

from os import remove
from os.path import isfile
from datetime import datetime as date
from multiprocessing.pool import ThreadPool

import config
from process import process_features

class DataRetriever():
    def __init__(self, cfg:config.cfg):
        self.cfg = cfg
        
        # Read data from files
        self.read_data()
    
    def read_data(self):
        
        self.ex_data = []
        if not self.cfg.force_data_reset:
            for n in self.cfg.nations:
                self.cfg.write_log(f"Data: Reading { n } data...", log.info)
                nation_data = {}
                for k in self.cfg.suffix_names.keys():
                    filename = self.cfg.data_prefix + n.lower() + self.cfg.suffix[k] + ".npy"
                    nation_data[k] = np.array(np.load(filename)).tolist() if isfile(filename) else []
                self.ex_data.append(nation_data)
            self.cfg.write_log("Data: Reading nation data complete.", log.info)
        else:
            self.cfg.write_log("Data: Force reset initiated...", log.warning, True)
            self.ex_data = [{ k : [] for k in self.cfg.suffix_names.keys() } for _ in self.cfg.nations]
            
            files = glob.glob(self.cfg.data_prefix + "*")
            for f in files:
                remove(f)

            self.cfg.write("force_data_reset", False)
    
    def retrieve_mining_data(self):
        self.cfg.write_log("Data: Constructing nation data...", log.info)
        for i, n in enumerate(self.cfg.nations):
                
            # Open the Google Sheets document
            sheet = self.cfg.ws.worksheet(n + " Data")
            
            # Find column for leyline screenshots
            leyline_col = sheet.find("Leyline", 1).col
            # Get all image URLs
            data_amount = len(sheet.col_values(leyline_col)[2:])
            # Check if current data is sufficent
            current_amount = len(self.ex_data[i]["d"])
            if data_amount == current_amount:
                self.cfg.write_log(f"Data: [{n}] Looks done already.", log.info)
                continue
            self.cfg.write_log(f"Data: [{n}] Starting data retrieval...", log.info)
            
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
            self.ex_data[i]['u'].extend(image_urls)
            
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

            self.cfg.write_log(f"Data: [{n}] Storing normal info...", log.info)

            # Save features and labels to .npy files
            for l in self.cfg.suffix_names.keys():
                if l != 'f':
                    np.save(self.cfg.data_prefix + n.lower() + self.cfg.suffix[l] + ".npy", np.array(self.ex_data[i][l]))

            self.cfg.write_log(f"Data: [{n}] Starting image processing...", log.info)
            # Iterate through each image URL
            self.retrieve_images(n, i, image_urls)
            #for url in progress.tqdm(image_urls, desc=n + " Image Progress", total=len(image_urls)):
                # Get image features from url
                #self.ex_data[i]['f'].append(process_features(url))
            # Save features to .npy file
            np.save(self.cfg.data_prefix + n.lower() + self.cfg.suffix['f'] + ".npy", np.array(self.ex_data[i]['f']))
                
            self.cfg.write_log(f"Data: [{n}] Data retrieved.", log.info)
            
        self.cfg.write_log("Data: Updating timestamp...", log.info, True)

        # Write the timestamp for when data was updated
        self.cfg.write_model("Data", date.now().timestamp())
            
        self.cfg.write_log("Data: Features and labels saved successfully.", log.info, True)

    def retrieve_images(self, region, region_num, urls):
        def thread_retrieve(url):
            return process_features(url)
        start = time.time()
        with ThreadPool(processes=100) as pool:
            for result in pool.map(thread_retrieve, urls):
                self.ex_data[region_num]['f'].append(result)
        print(f"Data: [{ region }] Images processed in { time.time() - start :.2f}s")

# Main function
def main():
    log.basicConfig(format="%(message)s", level=log.INFO)
    data = DataRetriever(True)
    data.retrieve_mining_data()

# Call the main function
if __name__ == "__main__":
    main()
