import requests
import threading
import webbrowser
import numpy as np
import tkinter as tk
import logging as log
from io import BytesIO
from tkinter import ttk, font
from functools import partial
from datetime import datetime
from PIL import Image, ImageTk

import config
import retrieve
from interface import Interface

class ErrorCheck():
    def __init__(self, face:Interface) -> None:
        self.face = face
        self.cfg = face.cfg
        self.input = face.input
        
        self.cfg.write_log("ErrorCheck: Retrieving data...", log.info)
        self.data = retrieve.DataRetriever(self.cfg)
        self.cfg.write_log("ErrorCheck: Initializing structure...", log.info)
        self.define_data()
    
    # Define variables that shown and interact with data
    def define_data(self):
        self.var = {}
        self.probs = {}
        self.order = {}
        self.leyline_max = {}
        self.nation_data = {}
        
        self.sheet_link = "https://docs.google.com/spreadsheets/d/{}/edit#gid={}".format(self.cfg.ws.id, self.cfg.ws.worksheet(f"Mondstadt Data").id)
        self.image_thread = None
        self.selected_error = 0
        self.selected_nation = tk.StringVar(value="Mondstadt")
        
        # Handle display of all error names
        for nation_index, n in enumerate(self.cfg.nations):
            self.nation_data[n] = self.data.ex_data[nation_index]
            # Get the confidence of each guess
            X_test = self.nation_data[n]['f']
            model = self.face.input.model_leyline[n].classifier
            self.probs[n] = np.min(np.array([np.max(estimator.predict_proba(X_test), axis=1) for estimator in model.estimators_]).T, axis=1)
            self.order[n] = np.argsort(self.probs[n])
            # Get the leyline number for the region
            self.leyline_max[n] = np.max(np.concatenate([self.nation_data[n]['y'], self.nation_data[n]['b']]))
        
        # Data display frame setup
        self.dataframe = ttk.Frame(self.face.root, padding="10", width=750)
        self.dataframe.grid(row=0, column=0, sticky="NSEW")
        
        # Display the leyline image
        self.dataframe_image = ttk.Frame(self.dataframe, padding="10")
        self.dataframe_image.pack(side="left", fill="both")
        self.leyline_im = ttk.Label(self.dataframe_image, image="")
        self.leyline_im.pack(fill="both")
        
        # Create a frame for text data
        self.dataframe_text = ttk.Frame(self.dataframe, padding="10", width=450)
        self.dataframe_text.pack(side="right", fill="both")
        
        # Create a frame for date and confidence
        self.dataframe_info = ttk.Frame(self.dataframe_text, width=450)
        self.dataframe_info.grid(row=0, column=0, sticky='NSEW')
        self.dataframe_info.columnconfigure(0, weight=5)
        self.dataframe_info.columnconfigure(1, weight=1)
        
        self.label_date = ttk.Label(self.dataframe_info, text="00/00/0000")
        self.label_date.grid(row=0, column=0, sticky="NW")
        self.label_conf = ttk.Label(self.dataframe_info, text="00.00%")
        self.label_conf.grid(row=0, column=1, sticky="NE")
        
        # Create a frame for leyline information
        self.dataframe_leyline = ttk.Frame(self.dataframe_text)
        self.dataframe_leyline.grid(row=1, column=0, sticky='NSEW')
        self.dataframe_leyline_max = max(self.leyline_max.values()) + 1
        self.leyline_labels = {}
        self.leyline_values = {}
        for i in range(0, self.dataframe_leyline_max):
            self.dataframe_leyline.columnconfigure(i, weight=1)
            self.leyline_labels[i] = ttk.Label(self.dataframe_leyline, text="X")
            self.leyline_labels[i].grid(row=0, column=i)
            self.leyline_values[i] = ttk.Label(self.dataframe_leyline, text="Y")
            self.leyline_values[i].grid(row=1, column=i)
        
        # Create a frame for links
        self.dataframe_links = ttk.Frame(self.dataframe_text, width=450)
        self.dataframe_links.grid(row=2, column=0, sticky='NSEW')
        self.dataframe_links.columnconfigure(0, weight=5)
        self.dataframe_links.columnconfigure(1, weight=1)
        
        self.link_url = self.create_hyperlink_label_show(self.dataframe_links, 0, 0, "link_url")
        self.link_url.grid_configure(sticky="SW")
        self.link_sheet = self.create_hyperlink_label_hide(self.dataframe_links, 0, 1, "link_sheet", "[>]")
        self.link_sheet.grid_configure(sticky="SE")
        
        # Create a frame for navigation
        self.dataframe_nav = ttk.Frame(self.dataframe_text)
        self.dataframe_nav.grid(row=3, column=0, sticky='SEW')
        [self.dataframe_nav.columnconfigure(c, weight=1) for c in range(0,4)]
        
        def on_previous():
            self.selected_error = (self.selected_error - 1) % len(self.probs[self.selected_nation.get()])
            self.change_error(self.order[self.selected_nation.get()][self.selected_error])
        self.nav_prev = ttk.Button(self.dataframe_nav, text="Previous", command=on_previous)
        self.nav_prev.grid(row=0, column=0, sticky="SW")
        
        def on_next():
            self.selected_error = (self.selected_error + 1) % len(self.probs[self.selected_nation.get()])
            self.change_error(self.order[self.selected_nation.get()][self.selected_error])
        self.nav_next = ttk.Button(self.dataframe_nav, text="Next", command=on_next)
        self.nav_next.grid(row=0, column=3, sticky="SE")
        
        # Button frame setup
        self.buttonframe = ttk.Frame(self.face.root, padding="10")
        self.buttonframe.grid(row=1, column=0, sticky='EWS')
        [self.buttonframe.columnconfigure(i, weight=1) for i in range(0,5)]
        
        # Nation selector
        # Whenever the combobox changed values
        def on_combo_selected(event):
            self.selected_error = 0
            self.sheet_link = "https://docs.google.com/spreadsheets/d/{}/edit#gid={}".format(self.cfg.ws.id, self.cfg.ws.worksheet(f"{self.selected_nation.get()} Data").id)
            self.change_error(self.order[self.selected_nation.get()][self.selected_error])
            
        self.nation_combo = ttk.Combobox(self.buttonframe, textvariable=self.selected_nation, values=self.cfg.nations, state="readonly")
        self.nation_combo.bind("<<ComboboxSelected>>", on_combo_selected)
        self.nation_combo.current(0)
        self.nation_combo.grid(row=1,column=0,sticky='SEW')
        
        self.change_error(self.order["Mondstadt"][0])
    
    def change_error(self, index):
        nation = self.selected_nation.get()
        nation_data = self.nation_data[nation]
        url = nation_data['u'][index]
        def get_image():
            response = requests.get(url, headers = {'User-agent': 'machasins'})
            image_data = Image.open(BytesIO(response.content))
            image_size = (int(image_data.size[0] * 200.0 / max(image_data.size)), int(image_data.size[1] * 200.0 / max(image_data.size)))
            new_image = ImageTk.PhotoImage(image_data.resize(image_size))
            self.leyline_im.configure(image=new_image)
            self.leyline_im.image = new_image
        self.image_thread = threading.Thread(target=get_image)
        self.image_thread.start()
        
        date = datetime(nation_data['d'][index][0], nation_data['d'][index][1], nation_data['d'][index][2])
        self.label_date.configure(text=date.strftime("%m/%d/%y"))
        self.label_conf.configure(text="{:.0%}".format(self.probs[nation][index]))
        [self.leyline_labels[i].configure(text="") for i in range(0, self.dataframe_leyline_max)]
        [self.leyline_values[i].configure(text="") for i in range(0, self.dataframe_leyline_max)]
        [self.leyline_labels[i - 1].configure(text=str(i)) for i in range(1, self.leyline_max[nation] + 2)]
        self.leyline_values[nation_data['y'][index]].configure(text="y")
        self.leyline_values[nation_data['b'][index]].configure(text="b")
        self.var["link_url"].set(url)
        date_diff = date - datetime(nation_data['d'][0][0], nation_data['d'][0][1], nation_data['d'][0][2])
        self.var["link_sheet"].set(f"{self.sheet_link}&range=A{3 + date_diff.days}")
        
    # Create a string variable and store it in a dictionary
    def create_str_var(self, name: str, starting_val: str = "") -> tk.StringVar:
        self.var[name] = tk.StringVar(value=starting_val)
        return self.var[name]
    
    # Create an int variable and store it in a dictionary
    def create_int_var(self, name: str, starting_val: int = 0) -> tk.IntVar:
        self.var[name] = tk.IntVar(value=starting_val)
        return self.var[name]
    
    # Create a hyperlink label that can be clicked on
    def create_hyperlink_label_show(self, frame:tk.Misc, row:int, col:int, index:str) -> ttk.Label:
        hyperlink = ttk.Label(frame, cursor='hand2', textvariable=self.create_str_var(index, "www.google.com"))
        hyperlink.grid(row=row,column=col)
        hyperlink.bind("<Button-1>", partial(open_browser, self.var, index))
        hyperlink.bind("<Enter>", partial(hover_font, hyperlink, 'blue', self.face.underlinefont))
        hyperlink.bind("<Leave>", partial(hover_font, hyperlink, 'black', self.face.defaultfont))
        return hyperlink
    
    # Create a hyperlink label that can be clicked on
    def create_hyperlink_label_hide(self, frame:tk.Misc, row:int, col:int, index:str, display_text:str) -> ttk.Label:
        hyperlink = ttk.Label(frame, cursor='hand2', text=display_text)
        hyperlink.grid(row=row,column=col)
        self.create_str_var(index, "www.google.com")
        hyperlink.bind("<Button-1>", partial(open_browser, self.var, index))
        hyperlink.bind("<Enter>", partial(hover_font, hyperlink, 'blue', self.face.underlinefont))
        hyperlink.bind("<Leave>", partial(hover_font, hyperlink, 'black', self.face.defaultfont))
        return hyperlink
    
    # Run the main loop for the application
    def main_loop(self):
        self.root.mainloop()


# Change font when hovered subfunction
def hover_font(widget, color, font, event):
    widget.configure(foreground=color, font=font)

# Open browser to URL subfunction
def open_browser(variables, index, event):
    webbrowser.open_new_tab(variables[index].get())

if __name__ == "__main__":
    cfg = config.cfg()
    i = ErrorCheck(cfg)
    i.main_loop()