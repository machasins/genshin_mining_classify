import numpy as np
import logging as log

from os.path import isfile
from joblib import dump, load
from datetime import datetime as date

import config
import process

class Trainer():
    def __init__(self, do_reset, cfg:config.cfg):
        self.cfg = cfg
        # Read model metadata
        self.model = {}
        categories = ["mining"]
        attributes = ["file", "accuracy"]
        for c in categories:
            self.model[c] = {}
            self.model[c][attributes[0]] = {}
            self.model[c][attributes[1]] = {}
            for n in self.cfg.nations:
                self.model[c][attributes[0]][n] = isfile(self.cfg.model_prefix + n.lower() + "_" + c + ".joblib")
                self.model[c][attributes[1]][n] = self.cfg.accuracy[c][n] or -1
        self.force_reset = do_reset
        self.confidence = 0
    
    def predict(self, features):
        return None
    
    # Update timestamp for config
    def update_timestamp(self, nation, suffix):
        self.cfg.write_model(nation + suffix, date.now().timestamp())
    
    # Update AI accuracy for config
    def update_accuracy(self, nation, suffix, accuracy):
        self.cfg.write_model(nation + suffix + "_accuracy", accuracy)

class MiningTrainer(Trainer):
    def __init__(self, nation, do_reset, cfg):
        super().__init__(do_reset, cfg)
        
        self.mining = self.model["mining"]
        
        if self.mining["file"][nation] and not self.force_reset:
            self.cfg.write_log(f"Mining: [{ nation }] Loading existing model...", log.info)
            self.classifier = load(self.cfg.model_prefix + nation + "_mining.joblib")
            self.accuracy = self.mining["accuracy"][nation]
        else:
            self.cfg.write_log(f"Mining: [{ nation }] Old mining model out of date, training new model...", log.warning, False)
            self.list_order = ['d', '1', 'y', 'b']
            lists = { k : np.load(self.cfg.data_prefix + nation.lower() + self.cfg.suffix[k] + ".npy") for k in self.list_order }
            feature = np.concatenate([
                lists['d'], 
                lists['1'], 
                lists['y'].reshape(-1, 1), 
                lists['b'].reshape(-1, 1)
                ], axis=1)
            self.cfg.write_log(f"Mining: [{ nation }] Loaded data...", log.info)
            labels = np.load(self.cfg.data_prefix + nation.lower() + self.cfg.suffix['2'] + ".npy")
            self.cfg.write_log(f"Mining: [{ nation }] Loaded results...", log.info)
            
            self.cfg.write_log(f"Mining: [{ nation }] Training...", log.info)
            self.classifier, self.accuracy = process.train_mrfc(feature, labels)
            self.cfg.write_log(f"Mining: [{ nation }] Training complete", log.info)
                
            dump(self.classifier, self.cfg.model_prefix + nation.lower() + "_mining.joblib")
            self.update_accuracy(nation, "_mining", self.accuracy)
        self.cfg.write_log("Mining: [" + nation + "] Loaded Mining AI with {0:.2%} accuracy.".format(self.accuracy), log.info)

    # Function to predict Leyline location in a given screenshot
    def predict(self, features):
        # Use the trained classifier to predict the Leyline location
        predicted_location = self.classifier.predict(np.array(features).reshape(1,-1))[0]
        # Update the confidence of the prediction
        self.confidence = np.average(np.array([np.max(estimator.predict_proba(np.array(features).reshape(1,-1)), axis=1) for estimator in self.classifier.estimators_]).T, axis=1)[0]
        # Return the predicted location
        return predicted_location