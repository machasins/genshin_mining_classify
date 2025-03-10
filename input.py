import re
import os
import json
import time
import sqlite3
import logging as log
from datetime import datetime as date

import train
import config
import retrieve

class Input():
    def __init__(self, cfg:config.cfg) -> None:
        self.cfg = cfg
        
        self.data_reset = self.cfg.force_data_reset or (date.now() - date.fromtimestamp(self.cfg.timestamp)).days >= self.cfg.data_days or cfg.arg.update
        if self.data_reset:
            self.cfg.write_log("Input: Reseting data...", log.info, True)
            retrieve.DataRetriever(cfg).retrieve_mining_data()
        
        self.cfg.write_log("Input: Retrieving AI models...", log.info)
        self.model_leyline = {}
        self.model_mining = {}
        if self.cfg.force_model_reset or self.data_reset:
            self.cfg.write_log("Input: Initialing model reset...", log.warning, True)
        for n in self.cfg.nations:
            self.model_leyline[n] = train.LeylineTrainer(n, self.cfg.force_model_reset or self.data_reset, cfg)
            self.model_mining[n] = train.MiningTrainer(n, self.cfg.force_model_reset or self.data_reset, cfg)
        if self.cfg.force_model_reset:
            data = {}
            with open("config.json", 'r') as d:
                data = json.load(d)
            data["force_model_reset"] = False
            with open("config.json", 'w') as d:
                json.dump(data, d, indent=2)
        
        self.cfg.write_log("Input: Connecting to SQL...", log.info)
        self.connect_sql()
        
        # Initialize cached data
        self.ore_shown = {}
        self.ore_hidden = {}
        self.leyline_url = {}
        self.leyline_class = {}
        
        # Initialize sheets data
        self.leyline_col = self.cfg.sheet.find("Leyline", 1).col
        self.ending_col = self.cfg.sheet.find("Options:", 1).col - 3
        
        self.cfg.write_log("Input: Ready", log.info)
        
        
    
    def connect_sql(self):
        # Connect to Ditto database
        self.db = sqlite3.connect(os.getenv("APPDATA") + "/Ditto/Ditto.db")
        self.db.create_function("REGEXP", 2, regexp)
        self.cur = self.db.cursor()
        
    def try_access_sheets(self, f):
        while True:
            try:
                ret = f()
                return ret
            except:
                time.sleep(60)
    
    # Get the current date, compared to the start date of the nation's data
    def get_date(self) -> list:
        def f():
            current_day = date.strptime(self.cfg.sheet.acell("AY4").value, '%m/%d/%y')
            # Return the difference in days between the current day and the start date
            self.date = [current_day.year, current_day.month, current_day.day]
            return self.date
        return self.try_access_sheets(f)
    
    # Get the shown ores (1)
    def get_ore_shown(self, nation: int) -> list:
        def f():
            # Get the relevant data
            row = (nation + 1) * 3
            data = self.cfg.sheet.range(row, 1, row, self.leyline_col - 1)
            # Extract integers from the data
            data = [(int(x.value) if re.search('[0-9]', x.value) else '') for x in data]
            # Get the index of all 1s in the data, if none return negative
            self.ore_shown[nation] = [i for i, x in enumerate(data) if x == 1] if 1 in data else [-2]
            return self.ore_shown[nation]
        return self.try_access_sheets(f)
    
    # Get the hidden ores (2)
    def get_ore_hidden(self, nation: int) -> list:
        def f():
            # Get the relevant data
            row = (nation + 1) * 3
            data = self.cfg.sheet.range(row, 1, row, self.leyline_col - 1)
            # Extract integers from the data
            data = [(int(x.value) if re.search('[0-9]', x.value) else '') for x in data]
            # Get the index of all 2s in the data, if none return negative
            self.ore_hidden[nation] = [i for i, x in enumerate(data) if x == 2] if 2 in data else [-2]
            return self.ore_hidden[nation]
        return self.try_access_sheets(f)
    
    # Write the hidden ores (2)
    def write_ore_hidden(self, nation: int) -> None:
        def f():
            # Update the relevant cells
            row = (nation + 1) * 3
            [self.cfg.sheet.update_cell(row, c + 1, '2') for c in self.ore_hidden[nation] if not self.cfg.sheet.cell(row, c + 1).value]
        return self.try_access_sheets(f)
    
    # Get the image URL for the leylines
    def get_leyline_url(self, nation: int) -> str:
        def f():
            self.leyline_url[nation] = self.cfg.sheet.cell((nation + 1) * 3, self.leyline_col).value
            return self.leyline_url[nation]
        return self.try_access_sheets(f)
    
    # Get the image URL for the leylines from the clipboard
    def get_leyline_url_data(self) -> list:
        # Poll the SQL DB for the last N image links copied
        urls = [i[0] for i in self.cur.execute(
            f"SELECT mText FROM Main WHERE mText REGEXP '^https://i\.imgur\.com/.*\.png$' ORDER BY lID DESC LIMIT { len(self.cfg.nations) }"
            ).fetchall()]
        # return the reversed url list
        self.urls = urls[::-1]
        return self.urls
    
    # Write the leyline url
    def write_leyline_url(self, nation):
        def f():
            # Update the google sheet with the aquired data
            self.cfg.sheet.update_cell((nation + 1) * 3, self.leyline_col, self.urls[nation])
        return self.try_access_sheets(f)
        
    # Get the leyline classifications
    def get_leyline_class(self, nation: int) -> list:
        def f():
            # Get the relevant data
            row = (nation + 1) * 3
            data = [x.value for x in self.cfg.sheet.range(row, self.leyline_col + 1, row, self.ending_col - 1)]
            # Get indexes of 'y' and 'b', otherwise return negative
            self.leyline_class[nation] = [data.index('y'), data.index('b')] if 'y' in data and 'b' in data else [-2, -2]
            return self.leyline_class[nation]
        return self.try_access_sheets(f)
    
    # Write the leyline classifications
    def write_leyline_class(self, nation: int) -> None:
        def f():
            # Update the relevant cells
            row = (nation + 1) * 3
            self.cfg.sheet.update_cell(row, self.leyline_col + self.leyline_class[nation][0] + 1, 'y')
            self.cfg.sheet.update_cell(row, self.leyline_col + self.leyline_class[nation][1] + 1, 'b')
        return self.try_access_sheets(f)
    
    # Format the shown ores in a displayable format
    def get_ore_shown_formatted(self, nation: int) -> str:
        return "[" + ", ".join(["{:2}".format(x + 1) for x in self.ore_shown[nation]]) + "]"
    
    # Format the hidden ores in a displayable format
    def get_ore_hidden_formatted(self, nation: int) -> str:
        return "[" + ", ".join(["{:2}".format(x + 1) for x in self.ore_hidden[nation]]) + "]"
    
    # Format the leylines in a displayable format
    def get_leyline_class_formatted(self, nation: int) -> str:
        return "[{:2}, {:2}]".format(self.leyline_class[nation][0] + 1, self.leyline_class[nation][1] + 1)

def regexp(expr, item):
    reg = re.compile(expr)
    return reg.search(item) is not None

if __name__ == "__main__":
    i = Input()
    i.set_leyline_url()