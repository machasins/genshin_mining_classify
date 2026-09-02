import re
import json
import time
import logging as log
from datetime import datetime as date

import train
import config
import retrieve

import gspread
from gspread.utils import rowcol_to_a1 as a1
from gspread.utils import ValueRenderOption, ValueInputOption

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
            self.model_mining[n] = train.MiningTrainer(n, self.cfg.force_model_reset or self.data_reset, cfg)
        if self.cfg.force_model_reset:
            data = {}
            with open("config.json", 'r') as d:
                data = json.load(d)
            data["force_model_reset"] = False
            with open("config.json", 'w') as d:
                json.dump(data, d, indent=2)
        
        # Initialize cached data
        self.ore_shown = {}
        self.ore_hidden = {}
        self.leyline_url = {}
        self.leyline_class = {}
        
        # Initialize sheets data
        self.leyline_col = self.cfg.sheet.find("Leylines", 1).col
        self.ending_col = self.cfg.sheet.find("Options:", 1).col - 3
        
        self.cfg.write_log("Input: Ready", log.info)
        
    def try_access_sheets(self, f):
        while True:
            try:
                ret = f()
                return ret
            except:
                time.sleep(10)
    
    # Get the current date, compared to the start date of the nation's data
    def get_date(self) -> list:
        def f():
            current_day = date.today()
            # Return the difference in days between the current day and the start date
            self.date = [current_day.year, current_day.month, current_day.day]
            return self.date
        return self.try_access_sheets(f)
    
    # Get the leyline classifications
    def get_leyline_class(self, nation: int) -> list:
        def f():
            # Get the relevant data
            row = (nation + 1) * 3
            data = [x.value for x in self.cfg.sheet.range(row, self.leyline_col, row, self.ending_col - 1)]
            # Get indexes of 'y' and 'b', otherwise return negative
            self.leyline_class[nation] = [data.index('y'), data.index('b')] if 'y' in data and 'b' in data else [-2, -2]
            return self.leyline_class[nation]
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
    
    def write_ore_hidden_all(self) -> None:
        updates: dict[tuple[int, int], (str, int)] = {}
        nations: list[int] = range(0, len(self.cfg.nations))
        for n in nations:
            if self.ore_hidden[n]:
                for h in self.ore_hidden[n]:
                    updates[((n + 1) * 3, h + 1)] = 2
        
        unique_rows = list(set([r for r, _ in updates.keys()]))
        max_col = max([coord[1] for coord in updates.keys()])
        
        ranges = [f"{ a1(r, 1) }:{ a1(r, max_col) }" for r in unique_rows]
        data: list[gspread.ValueRange] = self.cfg.sheet.batch_get(ranges, value_render_option=ValueRenderOption.unformatted)
        for (r, c), value in updates.items():
            row = unique_rows.index(r)
            if not data[row] or not data[row][0]:
                data[row] = [[]]
            if len(data[row]) < max_col:
                data[row][0].extend([None]*(max_col - len(data[row][0])))
                data[row][0] = [None if d == "" else d for d in data[row][0]]
            if data[row][0][c - 1] is None:
                data[row][0][c - 1] = value
        self.cfg.sheet.batch_update([{ "range" : r, "values" : data[i] } for i, r in enumerate(ranges)], value_input_option=ValueInputOption.user_entered)
    
    # Format the shown ores in a displayable format
    def get_ore_shown_formatted(self, nation: int) -> str:
        return "[" + ", ".join(["{:2}".format(x + 1) for x in self.ore_shown[nation]]) + "]"
    
    # Format the hidden ores in a displayable format
    def get_ore_hidden_formatted(self, nation: int) -> str:
        return "[" + ", ".join(["{:2}".format(x + 1) for x in self.ore_hidden[nation]]) + "]"
    
    # Format the leylines in a displayable format
    def get_leyline_class_formatted(self, nation: int) -> str:
        return "[{:2}, {:2}]".format(self.leyline_class[nation][0] + 1, self.leyline_class[nation][1] + 1)

if __name__ == "__main__":
    i = Input()