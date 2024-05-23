import sys
import json
import time
import webbrowser
import threading
import logging as log
from tkinter import *
from tkinter import ttk, font
from functools import partial
from distutils.util import strtobool

import input
import process
from process import write_log

class Interface():
    def __init__(self, verbose) -> None:
        write_log("Interface: Initializing input...", log.info, True, verbose, end="\r")
        self.input = input.Input(verbose)
        
        write_log("Interface: Reading config...", log.info, True, verbose, end="\r")
        self.read_config()
        write_log("Interface: Initializing structure...", log.info, True, verbose, end="\r")
        self.define_inital()
        write_log("Interface: Initializing data...", log.info, True, verbose, end="\r")
        self.define_data()
        write_log("Interface: Ready.", log.info, False, verbose)
    
    # Get data from the config file
    def read_config(self):
        with open("config.json") as d:
            self.nations = json.load(d)["nations"]
    
    # Define the initial window variables
    def define_inital(self):
        # Root setup
        self.root = Tk()
        self.root.title("Genshin Impact Leyline Classifier")
        self.root.geometry(str(800) + "x" + str(200))
        self.root.resizable(FALSE,FALSE)
        self.root.lift()
    
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)
        
        # Font setup
        self.defaultfont = font.nametofont("TkDefaultFont")
        self.defaultfont.configure(family="Courier", size=8)
        self.underlinefont = self.defaultfont.copy()
        self.underlinefont.configure(underline=True)
        
        # Variable setup
        self.var = {}
        self.thread_queue = []
        self.queue_num = 0
        self.queue_processing = 0
    
    # Define variables that shown and interact with data
    def define_data(self):
        # Data display frame setup
        self.dataframe = ttk.Frame(self.root, padding="10")
        self.dataframe.grid(row=0, column=0, sticky='NEW')
    
        [self.dataframe.columnconfigure(i, weight=1) for i in range(0,6)]
        
        # Holders for nation data
        for i, n in enumerate(self.nations):
            self.dataframe.rowconfigure(i, weight=1)
            ttk.Label(self.dataframe, textvariable=self.create_str_var(f"{n}_ore_1", "[__, __, __, __, __]")).grid(row=i,column=0,sticky='NW')
            ttk.Label(self.dataframe, textvariable=self.create_str_var(f"{n}_ore_2", "[__, __, __, __, __]")).grid(row=i,column=1,sticky='NW')
            ttk.Label(self.dataframe, textvariable=self.create_str_var(f"{n}_ore_g", "[__, __, __, __, __]"), foreground='gray').grid(row=i,column=2,sticky='NW')
            ttk.Label(self.dataframe, textvariable=self.create_str_var(f"{n}_ley_yb", "[__, __]")).grid(row=i,column=4,sticky='NE')
            ttk.Label(self.dataframe, textvariable=self.create_str_var(f"{n}_ley_g", "[__, __]"), foreground='gray').grid(row=i,column=5,sticky='NE')
            
            # Handling the hyperlink for the image
            hyperlink = ttk.Label(self.dataframe, cursor='hand2', textvariable=self.create_str_var(f"{n}_ley_im", f"https://i.imgur.com/XXXXXX{i}.png"))
            hyperlink.grid(row=i,column=3,sticky='NE')
            hyperlink.bind("<Button-1>", partial(open_browser, self.var, f"{n}_ley_im"))
            hyperlink.bind("<Enter>", partial(hover_font, hyperlink, 'blue', self.underlinefont))
            hyperlink.bind("<Leave>", partial(hover_font, hyperlink, 'black', self.defaultfont))
            
            # Setup variables for data handling
            self.var[f"{n}_data"] = False
            self.var[f"{n}_ley_w"] = False
            self.var[f"{n}_ore_w"] = False
        
        # Button frame setup
        self.buttonframe = ttk.Frame(self.root, padding="10")
        self.buttonframe.grid(row=1, column=0, sticky='EWS')
        
        # Buttons for retrieving data from sheet, polling the ais
        self.buttonframe.rowconfigure(1, weight=1)
        [self.buttonframe.columnconfigure(i, weight=1) for i in range(0,5)]
        ttk.Button(self.buttonframe, text="Autofill", command=self.autofill).grid(row=1,column=0,sticky='SEW')
        ttk.Button(self.buttonframe, text="Retrieve Data", command=self.retrieve_data).grid(row=1,column=1,sticky='SEW')
        ttk.Button(self.buttonframe, text="LeylineAI", command=self.poll_leyline_ai).grid(row=1,column=3,sticky='SEW')
        ttk.Button(self.buttonframe, text="MiningAI", command=self.poll_mining_ai).grid(row=1,column=4,sticky='SEW')
        
        # Buttons for setting the data polled from the ais to the sheet
        self.buttonframe.rowconfigure(0, weight=1)
        [self.buttonframe.columnconfigure(i, weight=1) for i in range(0,5)]
        ttk.Button(self.buttonframe, text="Write URLs", command=self.write_leyline_urls).grid(row=0,column=1,sticky='SEW')
        ttk.Button(self.buttonframe, text="Write Leyline", command=self.write_leyline).grid(row=0,column=3,sticky='SEW')
        ttk.Button(self.buttonframe, text="Write Mining", command=self.write_mining).grid(row=0,column=4,sticky='SEW')
    
    # Retrieve data from the Google Sheet
    def retrieve_data(self):
        def indiv_retrieve_data(self, nation, index):
            self.var[f"{nation}_ore_1"].set(self.input.get_ore_shown_formatted(index))
            self.var[f"{nation}_ore_2"].set(self.input.get_ore_hidden_formatted(index))
            self.var[f"{nation}_ley_im"].set(self.input.get_leyline_url(index))
            self.var[f"{nation}_ley_yb"].set(self.input.get_leyline_class_formatted(index))
            self.var[f"{nation}_data"] = self.var[f"{nation}_ley_im"] != ""
        
        self.start_thread_queue(indiv_retrieve_data, self.nations)
    
    # Retrieve URLs from clipboard and write them to the Google Sheet
    def write_leyline_urls(self):
        # Get the urls from the clipboard
        urls = self.input.get_leyline_url_data()
        def indiv_write_leyline_urls(self, nation, index):
            self.input.set_leyline_url(index, urls[index])
            self.var[f"{nation}_ley_im"].set(urls[index])
            self.var[f"{nation}_data"] = self.var[f"{nation}_ley_im"] != ""
                
        self.start_thread_queue(indiv_write_leyline_urls, self.nations)
    
    # Poll the AI to see where it thinks the leylines are located within an image
    def poll_leyline_ai(self):
        def indiv_poll_leyline_ai(self, nation, index):
            # Check if enough data has been recieved
            if self.var[f"{nation}_data"]:
                # Poll Leyline AI
                features = process.process_features(self.var[f"{nation}_ley_im"].get())
                result = self.input.model_leyline[nation].predict(features)
                result = [x + 1 for x in result]
                # Format and display result
                self.var[f"{nation}_ley_g"].set("[" + ", ".join("{:2}".format(x) for x in result) + "]")
                self.var[f"{nation}_ley_w"] = True
                
        self.start_thread_queue(indiv_poll_leyline_ai, self.nations)
    
    # Write leyline data recieved from AI to the Google Sheet
    def write_leyline(self):
        def indiv_write_leyline(self, nation, index):
            # Check if AI has been run
            if self.var[f"{nation}_ley_w"]:
                self.input.set_leyline_class(index, self.var[f"{nation}_ley_g"].get())
                self.var[f"{nation}_ley_yb"].set(self.var[f"{nation}_ley_g"].get())
        
        self.start_thread_queue(indiv_write_leyline, self.nations)
    
    # Poll the AI to see where it thinks the hidden ores are in the world
    def poll_mining_ai(self):
        def indiv_poll_mining_ai(self, nation, index):
            # Get relevant data
            date = self.input.get_date()
            ore_shown = self.input.get_ore_shown(index)
            leylines = self.input.get_leyline_class(index)
            # Check if enough data has been recieved
            if ore_shown[0] >= 0 and leylines[0] >= 0:
                # Poll Mining AI
                result = self.input.model_mining[nation].predict(date + ore_shown + leylines)
                result = [x + 1 for x in result]
                # Format and display result
                self.var[f"{nation}_ore_g"].set("[" + ", ".join("{:2}".format(x) for x in result) + "]")
                self.var[f"{nation}_ore_w"] = True
        
        self.start_thread_queue(indiv_poll_mining_ai, self.nations)
    
    # Write mining data recieved from AI to the Google sheet
    def write_mining(self):
        def indiv_write_mining(self, nation, index):
            # Check if AI has been run
            if self.var[f"{nation}_ore_w"]:
                self.input.set_ore_hidden(index, self.var[f"{nation}_ore_g"].get())
                self.var[f"{nation}_ore_2"].set(self.var[f"{nation}_ore_g"].get())
        
        self.start_thread_queue(indiv_write_mining, self.nations)
    
    # Run all operations sequentially
    def autofill(self):
        self.retrieve_data()
        self.write_leyline_urls()
        self.poll_leyline_ai()
        self.write_leyline()
        self.poll_mining_ai()
        self.write_mining()
    
    # Start a thread per entry in the enumerable
    def start_thread_queue(self, func, enumerable):
        def start_threads(self):
            for i, n in enumerate(enumerable):
                thr = threading.Thread(target=func, args=(self, n, i))
                self.thread_queue.append(thr)
                thr.start()
        
        thr = threading.Thread(target=self.wait_for_queue, args=(self.queue_num, start_threads))
        thr.start()
        self.queue_num += 1
    
    # Wait for all queue entries to finish
    def wait_for_queue(self, queue_num, callback):
        while self.queue_processing != queue_num:
            time.sleep(0.5)
        
        all_dead = False
        while not all_dead:
            time.sleep(0.5)
            all_dead = True
            for t in self.thread_queue:
                t.join(1)
                all_dead &= not t.is_alive()
        
        self.thread_queue = []
        callback(self)
        self.queue_processing = queue_num + 1
    
    # Create a string variable and store it in a dictionary
    def create_str_var(self, name: str, starting_val: str = "") -> StringVar:
        self.var[name] = StringVar(value=starting_val)
        return self.var[name]
    
    # Create an int variable and store it in a dictionary
    def create_int_var(self, name: str, starting_val: int = 0) -> IntVar:
        self.var[name] = IntVar(value=starting_val)
        return self.var[name]
    
    # Run the main loop for the application
    def main_loop(self):
        self.root.mainloop()

# Change font when hovered subfunction
def hover_font(widget, color, font, event):
    widget.configure(foreground=color, font=font)

# Open browser to URL subfunction
def open_browser(variables, index, event):
    webbrowser.open_new(variables[index].get())

if __name__ == "__main__":
    log.basicConfig(format="%(message)s", level=log.INFO)
    try:
        verbose = bool(strtobool(sys.argv[1]))
    except:
        verbose = False
    i = Interface(verbose)
    i.main_loop()