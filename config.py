import json
import gspread
import argparse
import logging as log
from os.path import isfile
from datetime import datetime as date

class cfg:
    suffix_names = {
        'd' : "date",
        '1' : "ore1",
        '2' : "ore2",
        'f' : "feature",
        'u' : "url",
        'y' : "ylabel",
        'b' : "blabel"
        }
    
    def __init__(self):
        self.parse_arguments()
        self.read()
        self.connect_to_sheets()
    
    def write_log(self, message, func, important = False):
        if important or self.arg.verbose:
            func(message)
    
    def parse_arguments(self):
        parser = argparse.ArgumentParser(prog="Leyline Classifier",
                                    description="Classifies Leylines and Mining Outcrops with the use of AI")
        parser.add_argument('-v', '--verbose', dest="verbose", action="store_const", default=False, const=True, help="whether console output should be verbose")
        parser.add_argument('-o', '--output', dest="output", default="warning", help="what type of output should be printed", type=str, choices=["debug", "info", "warning", "error"])
        parser.add_argument('-f', '--fileOutput', dest="file", default=None, help="what file the output should be printed to", type=str)
        parser.add_argument('-r', '--reset', action="store_const", default=False, const=True, help="whether to reset stored data")
        parser.add_argument('-u', '--update', action="store_const", default=False, const=True, help="whether to force update stored data")
        parser.add_argument('-m', '--model', action="store_const", default=False, const=True, help="whether to force the model to be trained again")
        parser.add_argument('-a', '--autofill', action="store_const", default=False, const=True, help="whether the program should run the autofill command at startup")
        parser.add_argument('-e', '--errorcheck', action="store_const", default=False, const=True, help="whether the program should test the data for errors")
        self.arg = parser.parse_args()
        log.basicConfig(format="%(message)s", level=self.arg.output.upper() if not self.arg.verbose else "INFO", filename=self.arg.file)
    
    def connect_to_sheets(self):
        if not isfile("key.json"):
            self.write_log("Input: File \"key.json\" does not exist. Please create the file with your Google Sheets credentials.", log.error, True)
            exit(1)
        # Authenticate Google Sheets API
        self.client = gspread.service_account("key.json")
        try:
            self.ws = self.client.open_by_key(self.sheet_id)
            self.sheet = self.ws.worksheet("DataEntry")
        except:
            self.write_log("Input: The sheet ID in the config is not valid with your account.", log.error, True)
            exit(1)
    
    def read(self):
        self.write_log("Config: Reading normal config...", log.info)
        with open("config.json") as config:
            self.data = json.load(config)
            self.force_data_reset = self.data["force_data_reset"] or self.arg.reset
            self.force_model_reset = self.data["force_model_reset"] or self.arg.reset or self.arg.model
            self.suffix = { n : self.data[v + "_suffix"] for n, v in cfg.suffix_names.items() }
            self.data_prefix = self.data["data_prefix"]
            self.model_prefix = self.data["model_prefix"]
            self.data_days = self.data["min_data_days"]
            self.sheet_id = self.data["sheet"]
            self.nations = self.data["nations"]
        self.write_log("Config: Reading model config...", log.info)
        try:
            with open(self.model_prefix + "config.json") as config:
                self.model_data = json.load(config)
        except FileNotFoundError:
            self.write_log("Config: Initializing model config file...", log.warning, True)
            with open(self.model_prefix + "config.json", "w") as d:
                self.model_data = {"Data" : date(0,0,0).timestamp}
                json.dump(self.model_data, d)
        self.timestamp = self.model_data["Data"]
        self.accuracy = {}
        self.accuracy["leyline"] = { n : self.model_data[n + "_leyline_accuracy"] for n in self.nations }
        self.accuracy["mining"] = { n : self.model_data[n + "_mining_accuracy"] for n in self.nations }
    
    def write(self, index, value):
        self.write_log(f"Config: Writing value { value } to { index } in normal config...", log.debug)
        self.data[index] = value
        with open("config.json", 'w') as config:
            json.dump(self.data, config, indent=2)
    
    def write_model(self, index, value):
        self.write_log(f"Config: Writing value { value } to { index } in model config...", log.debug)
        self.model_data[index] = value
        with open(self.model_prefix + "config.json", 'w') as config:
            json.dump(self.model_data, config, indent=2)