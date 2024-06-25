import json
import glob
import numpy as np
import logging as log

from os import remove
from os.path import isfile
from joblib import dump, load
from datetime import datetime as date

from retrieve import DataRetriever
import process
from process import write_log

class Trainer():
    def __init__(self, do_reset):
        # Read normal config
        with open("config.json") as config:
            self.data = json.load(config)
            self.nations = self.data["nations"]
            self.data_prefix = self.data["data_prefix"]
            self.model_prefix = self.data["model_prefix"]
            self.data_days = self.data["min_data_days"]
            self.suffix = {}
            self.suffix['l'] = self.data["label_suffix"]
            self.suffix['d'] = self.data["date_suffix"]
            self.suffix['1'] = self.data["ore1_suffix"]
            self.suffix['2'] = self.data["ore2_suffix"]
            self.suffix['f'] = self.data["feature_suffix"]
            self.suffix['y'] = self.data["ylabel_suffix"]
            self.suffix['b'] = self.data["blabel_suffix"]
        # Read model metadata
        with open(self.model_prefix + "config.json", "r") as config:
            self.model_data = json.load(config)
                        
            self.model = {}
            categories = ["leyline", "location", "mining"]
            attributes = ["file", "accuracy"]
            for c in categories:
                self.model[c] = {}
                self.model[c][attributes[0]] = {}
                self.model[c][attributes[1]] = {}
                for n in self.nations:
                    self.model[c][attributes[0]][n] = isfile(self.model_prefix + n.lower() + "_" + c + ".joblib")
                    self.model[c][attributes[1]][n] = self.model_data[n + "_" + c + "_accuracy"] if n + "_" + c + "_accuracy" in self.model_data else -1
        self.force_reset = do_reset
    
    def predict(self, features):
        return None
    
    # Update timestamp for config
    def update_timestamp(self, nation, suffix):
        with open(self.model_prefix + "config.json", 'w') as config:
            self.model_data[nation + suffix] = date.now().timestamp()
            json.dump(self.model_data, config, indent=2)
    
    # Update AI accuracy for config
    def update_accuracy(self, nation, suffix, accuracy):
        with open(self.model_prefix + "config.json", 'w') as config:
            self.model_data[nation + suffix + "_accuracy"] = accuracy
            json.dump(self.model_data, config, indent=2)
    
class LeylineTrainer(Trainer):
    def __init__(self, nation, do_reset, verbose):
        super().__init__(do_reset)
        
        self.leyline = self.model["leyline"]
        
        if isfile(self.model_prefix + nation + "_y.joblib") and not self.force_reset:
            self.classifier = {
                "y" : load(self.model_prefix + nation + "_y.joblib"),
                "b" : load(self.model_prefix + nation + "_b.joblib")
            }
            self.accuracy = self.leyline["accuracy"][nation] if self.leyline["accuracy"][nation] is list else [ -1, -1 ]
        else:
            write_log(f"Training: Old leyline model out of date, training new model for { nation }...", log.warning, False, verbose)
            feature = np.load(self.data_prefix + nation.lower() + self.suffix["f"] + ".npy")
            write_log(f"Training: Loaded data for { nation }...", log.info, True, verbose)
            ylabels = np.load(self.data_prefix + nation.lower() + self.suffix["y"] + ".npy")
            blabels = np.load(self.data_prefix + nation.lower() + self.suffix["b"] + ".npy")
            write_log(f"Training: Loaded results for { nation }...", log.info, True, verbose)
            
            write_log(f"Training: Training { nation }...", log.info, True, verbose)
            yclassifier, yaccuracy = process.train_svc(feature, ylabels)
            bclassifier, baccuracy = process.train_svc(feature, blabels)
            self.classifier = { "y" : yclassifier, "b" : bclassifier }
            self.accuracy = [ yaccuracy, baccuracy ]
            write_log(f"Training: Training { nation } complete", log.info, True, verbose)
                
            dump(self.classifier["y"], self.model_prefix + nation.lower() + "_y.joblib")
            dump(self.classifier["b"], self.model_prefix + nation.lower() + "_b.joblib")
            self.update_accuracy(nation, "_leyline", [yaccuracy, baccuracy])
        write_log("Loaded " + nation + " Leyline AI with {0:.2%} yellow accuracy and {1:.2%} blue accuracy.".format(self.accuracy[0], self.accuracy[1]), log.info, True, verbose)

    # Function to predict Leyline location in a given screenshot
    def predict(self, features):
        # Use the trained classifier to predict the Leyline location
        predicted_location = [self.classifier["y"].predict(features.reshape(1,-1))[0], self.classifier["b"].predict(features.reshape(1,-1))[0]]
        # Return the predicted location
        return predicted_location

class LocationTrainer(Trainer):
    def __init__(self, nation, do_reset, verbose):
        super().__init__(do_reset)
        
        self.location = self.model["location"]
        
        if self.location["file"][nation] and not self.force_reset:
            self.classifier = load(self.model_prefix + nation + "_location.joblib")
            self.accuracy = self.location["accuracy"][nation]
        else:
            write_log(f"Training: Old location model out of date, training new model for { nation }...", log.warning, False, verbose)
            self.list_order = ['d', 'y', 'b']
            lists = { k : np.load(self.data_prefix + nation.lower() + self.suffix[k] + ".npy") for k in self.list_order }
            feature = np.concatenate([
                lists['d'],
                lists['y'].reshape(-1, 1), 
                lists['b'].reshape(-1, 1)
                ], axis=1)
            labels = np.load(self.data_prefix + nation.lower() + self.suffix['1'] + ".npy")
            
            write_log(f"Training: Training { nation }...", log.info, True, verbose)
            self.classifier, self.accuracy = process.train_mrfc(feature, labels)
            write_log(f"Training: Training { nation } complete", log.info, True, verbose)
                
            dump(self.classifier, self.model_prefix + nation.lower() + "_location.joblib")
            self.update_accuracy(nation, "_location", self.accuracy)
        write_log("Loaded " + nation + " Location AI with {0:.2%} accuracy.".format(self.accuracy), log.info, True, verbose)

    # Function to predict Leyline location in a given screenshot
    def predict(self, features):
        # Use the trained classifier to predict the Leyline location
        predicted_location = self.classifier.predict(np.array(features).reshape(1,-1))[0]
        # Return the predicted location
        return predicted_location

class MiningTrainer(Trainer):
    def __init__(self, nation, do_reset, verbose):
        super().__init__(do_reset)
        
        self.mining = self.model["mining"]
        
        if self.mining["file"][nation] and not self.force_reset:
            self.classifier = load(self.model_prefix + nation + "_mining.joblib")
            self.accuracy = self.mining["accuracy"][nation]
        else:
            write_log(f"Training: Old mining model out of date, training new model for { nation }...", log.warning, False, verbose)
            self.list_order = ['d', '1', 'y', 'b']
            lists = { k : np.load(self.data_prefix + nation.lower() + self.suffix[k] + ".npy") for k in self.list_order }
            feature = np.concatenate([
                lists['d'], 
                lists['1'], 
                lists['y'].reshape(-1, 1), 
                lists['b'].reshape(-1, 1)
                ], axis=1)
            write_log(f"Training: Loaded data for { nation }...", log.info, True, verbose)
            labels = np.load(self.data_prefix + nation.lower() + self.suffix['2'] + ".npy")
            write_log(f"Training: Loaded results for { nation }...", log.info, True, verbose)
            
            write_log(f"Training: Training { nation }...", log.info, True, verbose)
            self.classifier, self.accuracy = process.train_mrfc(feature, labels)
            write_log(f"Training: Training { nation } complete", log.info, True, verbose)
                
            dump(self.classifier, self.model_prefix + nation.lower() + "_mining.joblib")
            self.update_accuracy(nation, "_mining", self.accuracy)
        write_log("Loaded " + nation + " Mining AI with {0:.2%} accuracy.".format(self.accuracy), log.info, True, verbose)

    # Function to predict Leyline location in a given screenshot
    def predict(self, features):
        # Use the trained classifier to predict the Leyline location
        predicted_location = self.classifier.predict(np.array(features).reshape(1,-1))[0]
        # Return the predicted location
        return predicted_location