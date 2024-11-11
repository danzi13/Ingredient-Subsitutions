import os
import json
import re
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.metrics.pairwise import euclidean_distances
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import numpy as np
import networkx as nx
import plotly.graph_objs as go
from collections import Counter
import random

# Load the JSON files
categories = ['leafy_green.json', 'red_meat.json', 'pasta.json']

FOOD_JSON_DIR = 'food-JSON'
graphs = {}


FL_OZ_TO_ML = 29.5735  # 1 fl oz = 29.5735 ml
STANDARD_SERVING_SIZE_ML = 240  # Standard serving size (240 mL or 1 cup)

all_products = []

def extract_serving_container(servings_text):
    match = re.search(r'(\d+(\.\d+)?)\s?servings', servings_text.lower())
    if match:
        return float(match.group(1))
    return 1.0  

def convert_to_grams(value):
    if value is None:
        return 0.0
    value = value.lower()
    if 'mg' in value:
        return float(re.sub(r'[^\d.]', '', value)) 
    elif 'mcg' in value:
        return float(re.sub(r'[^\d.]', '', value)) 
    else:
        return float(re.sub(r'[^\d.]', '', value)) 

def get_serving_size(serving_size_text):
    serving_size_text = serving_size_text.lower()

    # Search for serving size value in parentheses for cups first (e.g., "(1 ½ cup)", "(240ml)", "(8 fl oz)", or "(2 oz)")
    match_cups = re.search(r'\((\d+(\s+\d+)?\s*cups?)\)', serving_size_text)
    match_oz = re.search(r'\((\d+(\.\d+)?)(ml|fl oz|oz)\)', serving_size_text)

    if match_cups:
        # Extract the cups value
        cups_value = match_cups.group(1)
        # Convert cups to ml (1 cup = 240 ml)
        if '½' in cups_value:
            return 1.5 * 240  # Handle the 1 ½ cup case
        else:
            return float(cups_value.split()[0]) * 240  # Convert whole cups to ml

    elif match_oz:
        # Extract the numeric value and the unit (ml, fl oz, or oz)
        size_value = float(match_oz.group(1))
        unit = match_oz.group(3)

        # Convert to ml if necessary
        if 'fl oz' in unit or 'oz' in unit:
            return size_value * FL_OZ_TO_ML  # Converts fl oz or oz to ml
        else:
            return size_value  # Already in ml

    # If no valid serving size is found, default to standard serving size
    return STANDARD_SERVING_SIZE_ML


for category in categories:
    graph = nx.Graph()
    file_path = os.path.join(FOOD_JSON_DIR, category)
    with open(file_path, 'r') as file:
        products = json.load(file)
        for product in products:
            name = product["Product Name"]
            # Extract and normalize nutrients
            calories = float(product["Calories"]) if product["Calories"] else 0.0
            total_fat = convert_to_grams(product["Fat"]["Total Fat"])
            saturated_fat = convert_to_grams(product["Fat"]["Saturated Fat"])
            trans_fat = convert_to_grams(product["Fat"]["Trans Fat"])
            sodium = convert_to_grams(product["Sodium"])
            total_carbs = convert_to_grams(product["Carbs"]["Total Carbs"])
            fiber = convert_to_grams(product["Carbs"]["Fiber"])
            sugars = convert_to_grams(product["Carbs"]["Sugars"])
            protein = convert_to_grams(product["Protein"])
            
            # Get serving size and calculate scaling factor
            serving_size_text = product["Servings"]
            count = extract_serving_container(product["Servings"])
            serving_size_ml = get_serving_size(serving_size_text)  # Extract serving size in ml (convert if needed)
                
            scaling_factor = STANDARD_SERVING_SIZE_ML / serving_size_ml

            #extract and calculate price per serving size 
            price = float(product['Price'].replace('$', '').strip())  
            serving_size = extract_serving_container(serving_size_text) 
            price_per_container = price / serving_size
            product['Price'] = f"${price_per_container:.2f}"
            
            if not scaling_factor:
                print(scaling_factor)
                raise Exception
            
            print(count)
            graph.add_node(name, 
                           calories=calories,
                           total_fat=total_fat,
                           saturated_fat=saturated_fat,
                           trans_fat=trans_fat,
                           sodium=sodium,
                           total_carbs=total_carbs,
                           fiber=fiber,
                           sugars=sugars,
                           protein=protein,
                           price=price,
                           servings = count)
    graphs[category] = graph

import os
import json
import numpy as np
import networkx as nx
import random


# Ideal macros definition
ideal_macros = {
    'calories': 600,
    'total_fat': 20,
    'saturated_fat': 5,
    'trans_fat': 0,
    'sodium': 500,
    'total_carbs': 60,
    'fiber': 8,
    'sugars': 10,
    'protein': 25
}

import numpy as np
import random
import networkx as nx
import random
import numpy as np
import networkx as nx

# Define ideal macros
ideal_macros = {
    'calories': 600,
    'total_fat': 20,
    'saturated_fat': 5,
    'trans_fat': 0,
    'sodium': 500,
    'total_carbs': 60,
    'fiber': 8,
    'sugars': 10,
    'protein': 25
}

# Function to calculate the combined nutrition for selected nodes
def calculate_combined_nutrition(nodes, graphs):
    combined_nutrition = {key: 0 for key in ideal_macros.keys()}
    for graph, node in zip(graphs.values(), nodes):
        for key in ideal_macros.keys():
            combined_nutrition[key] += graph.nodes[node][key]
    return combined_nutrition

# Function to calculate reward based on distance to ideal macros
def calculate_reward(nutrition, ideal_macros):
    return -np.sqrt(sum((nutrition[key] - ideal_macros[key])**2 for key in ideal_macros))

# Reinforced optimization function to find the best nodes
def reinforced_optimization(graphs, ideal_macros, iterations):
    """
    Optimize the selection of one node from each graph to match ideal macros without scaling servings.
    """
    # Initialize best nodes with random selections from each graph
    current_nodes = [random.choice(list(graph.nodes)) for graph in graphs.values()]
    
    best_nodes = current_nodes
    best_nutrition = calculate_combined_nutrition(current_nodes, graphs)
    best_reward = calculate_reward(best_nutrition, ideal_macros)

    print("Starting Reinforced Optimization...")
    for _ in range(iterations):
        for i, graph in enumerate(graphs.values()):
            current_node = current_nodes[i]
            neighbors = list(graph.neighbors(current_node))

            if not neighbors:
                continue

            for neighbor in neighbors:
                trial_nodes = current_nodes.copy()
                trial_nodes[i] = neighbor

                trial_nutrition = calculate_combined_nutrition(trial_nodes, graphs)
                trial_reward = calculate_reward(trial_nutrition, ideal_macros)

                if trial_reward > best_reward:
                    best_nodes = trial_nodes
                    best_nutrition = trial_nutrition
                    best_reward = trial_reward

        # Print progress and results
        #print(f"Best Reward: {best_reward}")
        #print(f"Best Nutrition: {best_nutrition}")

    return best_nodes, best_nutrition

# Example graph structure (adjust as necessary for your data

# Assuming the nodes are already populated in the graphs with attributes like calories, fat, protein, etc.

# Perform reinforced optimization
best_nodes, best_nutrition = reinforced_optimization(graphs, ideal_macros, 100000)

# Output selected nodes and their servings
print("Selected nodes with servings:")
for node in best_nodes:
    # Assuming we have access to the servings value directly in the graph
    for graph in graphs.values():
        if node in graph.nodes:
            servings = graph.nodes[node].get("servings", "Not Available")
            print(f"Node: {node}, Servings: {servings}")

print("\nCombined nutrition:")
print(best_nutrition)

print("\nDifference from ideal macros:")
for key in ideal_macros:
    difference = best_nutrition[key] - ideal_macros[key]
    print(f"{key}: {difference}")