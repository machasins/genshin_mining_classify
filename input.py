import re
import os
import json
import gspread
import sqlite3
import logging as log
from os.path import isfile
from datetime import datetime as date

import train
import retrieve
from process import write_log

class Input():
    def __init__(self, verbose) -> None:
        write_log("Input: Reading configs...", log.info, True, verbose, end="\r")
        # Read config information
        data = {}
        with open("config.json") as d:
            data = json.load(d)
            self.sheet_id = data["sheet"]
            self.nations = data["nations"]
            self.data_days = data["min_data_days"]
            self.model_prefix = data["model_prefix"]
            self.force_data_reset = data["force_data_reset"]
            self.force_model_reset = data["force_model_reset"]
        try:
            with open(self.model_prefix + "config.json", "r") as d:
                self.data_time = date.fromtimestamp(json.load(d)["Data"])
        except FileNotFoundError:
            with open(self.model_prefix + "config.json", "w") as d:
                self.data_time = date(0,0,0)
                json.dump({"Data" : self.data_time.timestamp}, d)
        
        self.data_reset = self.force_data_reset or (date.now() - self.data_time).days >= self.data_days
        if self.data_reset:
            write_log("Input: Reseting data...", log.info, False, verbose)
            retrieve.DataRetriever(verbose).retrieve_mining_data()
        
        write_log("Input: Retrieving AI models...", log.info, True, verbose, end="\r")
        self.model_leyline = {}
        self.model_mining = {}
        for n in self.nations:
            self.model_leyline[n] = train.LeylineTrainer(n, self.force_model_reset or self.data_reset, verbose)
            self.model_mining[n] = train.MiningTrainer(n, self.force_model_reset or self.data_reset, verbose)
        if self.force_model_reset:
            data = {}
            with open("config.json", 'r') as d:
                data = json.load(d)
            data["force_data_reset"] = False
            with open("config.json", 'w') as d:
                json.dump(data, d, indent=2)
        
        write_log("Input: Connecting to SQL...", log.info, True, verbose, end="\r")
        self.connect_sql()
        write_log("Input: Connecting to Sheets...", log.info, True, verbose, end="\r")
        self.connect_sheet()
        write_log("Input: Ready", log.info, True, verbose)
        
    
    def connect_sql(self):
        # Connect to Ditto database
        self.db = sqlite3.connect(os.getenv("APPDATA") + "/Ditto/Ditto.db")
        self.db.create_function("REGEXP", 2, regexp)
        self.cur = self.db.cursor()
        
    def connect_sheet(self):
        if not isfile("key.json"):
            write_log("File \"key.json\" does not exist. Please create the file with your Google Sheets credentials.", log.error, False, False)
            exit(1)
        # Authenticate Google Sheets API
        self.client = gspread.service_account("key.json")
        try:
            self.ws = self.client.open_by_key(self.sheet_id)
            self.sheet = self.ws.worksheet("DataEntry")
        except:
            write_log("The sheet ID in the config is not valid with your account.", log.error, False, False)
            exit(1)
        
        self.leyline_col = self.sheet.find("Leyline", 1).col
        self.ending_col = self.sheet.find("Options:", 1).col - 3
    
    # Get the current date, compared to the start date of the nation's data
    def get_date(self) -> list:
        current_day = date.strptime(self.sheet.acell("AY4").value, '%m/%d/%y')
        # Return the difference in days between the current day and the start date
        return [current_day.year, current_day.month, current_day.day]
    
    # Get the shown ores (1)
    def get_ore_shown(self, nation: int) -> list:
        # Get the relevant data
        row = (nation + 1) * 3
        data = self.sheet.range(row, 1, row, self.leyline_col - 1)
        # Extract integers from the data
        data = [(int(x.value) if re.search('[0-9]', x.value) else '') for x in data]
        # Get the index of all 1s in the data, if none return negative
        return [i for i, x in enumerate(data) if x == 1] if 1 in data else [-2]
    
    # Get the hidden ores (2)
    def get_ore_hidden(self, nation: int) -> list:
        # Get the relevant data
        row = (nation + 1) * 3
        data = self.sheet.range(row, 1, row, self.leyline_col - 1)
        # Extract integers from the data
        data = [(int(x.value) if re.search('[0-9]', x.value) else '') for x in data]
        # Get the index of all 2s in the data, if none return negative
        return [i for i, x in enumerate(data) if x == 2] if 2 in data else [-2]
    
    # Set the hidden ores (2)
    def set_ore_hidden(self, nation: int, data: str) -> None:
        # Get integers from the formatted string
        ores = [int(i) for i in re.findall(r'\d+', data)]
        # Update the relevant cells
        row = (nation + 1) * 3
        [self.sheet.update_cell(row, c, '2') for c in ores]
    
    # Get the image URL for the leylines
    def get_leyline_url(self, nation: int) -> str:
        return self.sheet.cell((nation + 1) * 3, self.leyline_col).value
    
    # Get the image URL for the leylines from the clipboard
    def get_leyline_url_data(self) -> list:
        # Poll the SQL DB for the last N image links copied
        urls = [i[0] for i in self.cur.execute(
            f"SELECT mText FROM Main WHERE mText REGEXP '^https://i\.imgur\.com/.*\.png$' ORDER BY lID DESC LIMIT { len(self.nations) }"
            ).fetchall()]
        # return the reversed url list
        return urls[::-1]
    
    # Set the leyline url
    def set_leyline_url(self, nation, url):
        # Update the google sheet with the aquired data
        self.sheet.update_cell((nation + 1) * 3, self.leyline_col, url)
        
    # Get the leyline classifications
    def get_leyline_class(self, nation: int) -> list:
        # Get the relevant data
        row = (nation + 1) * 3
        data = [x.value for x in self.sheet.range(row, self.leyline_col + 1, row, self.ending_col - 1)]
        # Get indexes of 'y' and 'b', otherwise return negative
        return [data.index('y'), data.index('b')] if 'y' in data and 'b' in data else [-2, -2]
    
    # Set the leyline classifications
    def set_leyline_class(self, nation: int, data: str) -> None:
        # Get integers from the formatted string
        leylines = [int(i) for i in re.findall(r'\d+', data)]
        # Update the relevant cells
        row = (nation + 1) * 3
        self.sheet.update_cell(row, self.leyline_col + leylines[0], 'y')
        self.sheet.update_cell(row, self.leyline_col + leylines[1], 'b')
    
    # Format the shown ores in a displayable format
    def get_ore_shown_formatted(self, nation: int) -> str:
        return "[" + ", ".join(["{:2}".format(x + 1) for x in self.get_ore_shown(nation)]) + "]"
    
    # Format the hidden ores in a displayable format
    def get_ore_hidden_formatted(self, nation: int) -> str:
        return "[" + ", ".join(["{:2}".format(x + 1) for x in self.get_ore_hidden(nation)]) + "]"
    
    # Format the leylines in a displayable format
    def get_leyline_class_formatted(self, nation: int) -> str:
        data = self.get_leyline_class(nation)
        return "[{:2}, {:2}]".format(data[0] + 1, data[1] + 1)

def regexp(expr, item):
    reg = re.compile(expr)
    return reg.search(item) is not None

if __name__ == "__main__":
    i = Input()
    i.set_leyline_url()