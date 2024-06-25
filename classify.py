# Import necessary libraries
import logging as log
import numpy as np

import process
import train

# Define functions
# Consult nation AI to determine what nation image is in
def determine_nation(features: np.ndarray) -> str:
    nt = train.NationTrainer()
    nation = nt.predict(features)
    return nt.nations[nation]

# Consult leyline AI to determine what positions the leylines in the image are
def determine_leyline_positions(features: np.ndarray, nation: str) -> list:
    lt = train.LeylineTrainer(nation)
    positions = lt.predict(features)
    return [x + 1 for x in positions]

# Consult mining AI to determine what positions the outcrops are
def determine_mining_positions(features: np.ndarray, nation: str) -> list:
    mt = train.MiningTrainer(nation)
    positions = mt.predict(features)
    return [x + 1 for x in positions]

def classify_image_from_nation(url: str, nation: str) -> list:
    log.debug(f"Image to classify: { url }")
    
    features = process.process_features(url)
    log.debug("Image processed.")
    
    log.debug(f"Consulting { nation } Leyline AI...")
    positions = determine_leyline_positions(features, nation)
    print(f"Determined image displays yellow leyline at position { positions[0] } and blue leyline at position { positions[1] }.")
    return positions

def classify_mining_from_nation(data: np.ndarray, nation:str) -> list:
    log.debug(f"Consulting { nation } Mining AI...")
    positions = determine_mining_positions(data, nation)
    print(f"Determined mining outcrops are most likely at: { positions }")
    return positions
    
# Main function
def main():
    t = train.Trainer()
    urls = [
        "https://i.imgur.com/XDJqzWF.png", # Mondstadt, 7y 1b
        "https://i.imgur.com/q9GYaHM.png", # Liyue, 3y, 4b
        "https://i.imgur.com/yzU67PA.png", # Inazuma, 5y, 2b
        "https://i.imgur.com/dEhaRoj.png", # Sumeru, 6y, 7b
        "https://i.imgur.com/9iEpey1.png"  # Fontaine, 1y, 5b
        ]
    
    for u, n in zip(urls, t.nations):
        print("--------------------")
        classify_image_from_nation(u, n)
    print("--------------------")
    
    data_m = [
        [2024, 3, 14, 4, 6, 8, 5, 6], # Mondstadt, [1, 2, 7]
        [2024, 1, 23, 16, 18, 24, 25, 26, 11, 12], # Liyue, [17, 19, 20, 21, 23]
        [2024, 3, 4, 7, 9, 10, 12, 1, 2], # Inazuma, [2, 5, 8, 11]
        [2024, 3, 18, 1, 5, 31, 32, 8, 9], # Sumeru, [2, 3, 6, 8]
        [2024, 3, 23, 17, 18, 19, 20, 1, 5] # Fontaine, [4, 7, 9, 10]
        ]
    
    for d, n in zip(data_m, t.nations):
        print("--------------------")
        classify_mining_from_nation(d, n)
    print("--------------------")

# Call the main function
if __name__ == "__main__":
    log.basicConfig(format="%(message)s", level=log.DEBUG)
    main()