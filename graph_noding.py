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


# Load the JSON file
with open('milk.json', 'r') as file:
    products = json.load(file)

start_product = "365 by Whole Foods Market, Milk Whole Organic, 128 Fl Oz"

FL_OZ_TO_ML = 29.5735  # 1 fl oz = 29.5735 ml
STANDARD_SERVING_SIZE_ML = 240  # Standard serving size (240 mL or 1 cup)

# Function to convert nutrient values to grams (g)
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

# Function to extract and normalize serving size based on fl oz or mL
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


def extract_serving_container(servings_text):
    match = re.search(r'(\d+(\.\d+)?)\s?servings', servings_text.lower())
    if match:
        return float(match.group(1))
    return 1.0  

product_names = []
nutrient_data = []

for product in products:
    
    product_names.append(product["Product Name"])
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
    serving_size_ml = get_serving_size(serving_size_text)  # Extract serving size in ml (convert if needed)
        
    scaling_factor = STANDARD_SERVING_SIZE_ML / serving_size_ml

    #extract and calculate price per serving size 
    price = float(product['Price'].replace('$', '').strip())  
    serving_size = extract_serving_container(serving_size_text) 
    price_per_container = price / serving_size
    product['Price'] = f"${price_per_container:.2f}"
    
    
    # Normalize all values 
    nutrient_data.append([
        calories * scaling_factor,
        total_fat * scaling_factor,
        saturated_fat * scaling_factor,
        trans_fat * scaling_factor,
        sodium * scaling_factor,
        total_carbs * scaling_factor,
        fiber * scaling_factor,
        sugars * scaling_factor,
        protein * scaling_factor
    ])


# DF for easier handling
columns = ['Calories', 'Total Fat', 'Saturated Fat', 'Trans Fat', 'Sodium', 'Total Carbs', 'Fiber', 'Sugars', 'Protein']
df = pd.DataFrame(nutrient_data, columns=columns, index=product_names)

# Missing values are 0
df.fillna(0, inplace=True)

# Normalized nutrient data 
product_names = [product["Product Name"] for product in products]
product_prices = []
serving_sizes = []


# Extract serving sizes and prices
for product in products:
    price = float(product['Price'].replace('$', '').strip())
    product_prices.append(price)

price_scaler = StandardScaler()
normalized_prices = price_scaler.fit_transform(np.array(product_prices).reshape(-1, 1)).flatten()

# Scale the nutrient data 
scaler = StandardScaler()
scaled_data = scaler.fit_transform(df)

#distances between products based on the nutrient data
distances = euclidean_distances(scaled_data)

# Initialize a graph
G = nx.Graph()

for i in range(len(product_names)):
    G.add_node(product_names[i], 
               calories=nutrient_data[i][0],  #Calories
               total_fat=nutrient_data[i][1],  #Total Fat
               saturated_fat=nutrient_data[i][2],  #Saturated Fat
               trans_fat=nutrient_data[i][3],  #Trans Fat
               sodium=nutrient_data[i][4],  #Sodium
               total_carbs=nutrient_data[i][5],  #Total Carbs
               fiber=nutrient_data[i][6],  #Fiber
               sugars=nutrient_data[i][7],  #Sugars
               protein=nutrient_data[i][8], #Protein
               price=normalized_prices[i]) #Price

#Need to balance nearest neighbors
k = 3

for i in range(len(product_names)):
    nearest_neighbors = np.argsort(distances[i])[1:k+1]  

    for neighbor_index in nearest_neighbors:
        distance = distances[i, neighbor_index]  
                
        G.add_edge(product_names[i], product_names[neighbor_index], weight=distance)

# Biased walk
def biased_random_walk_with_weights(G, start_node, alpha, max_steps=100):
    """
    Perform a random walk with weighted probability for all neighbors.
    alpha = 1 only weight nutrients, alpha = 0 only weight costs, can be anywhere from 0 -> 1 depending on user goals

    See calculate_nutritional_score function for further bias for specific nutrient dense food
    
    """
    current_node = start_node
    visited_nodes = [current_node]

    for step in range(max_steps):
        neighbors = list(G.neighbors(current_node))
        
        current_price = G.nodes[current_node]['price']
        neighbor_prices = np.array([G.nodes[neighbor]['price'] for neighbor in neighbors])
        costs = current_price - neighbor_prices

        if np.std(costs) > 0:
            costs_norm = (costs - costs.min()) / (costs.max() - costs.min())
        else:
            costs_norm = np.zeros_like(costs)

        nutrition_scores = np.array([
            calculate_nutritional_score(
                G.nodes[neighbor].get('calories', 0),
                G.nodes[neighbor].get('total_fat', 0),
                G.nodes[neighbor].get('saturated_fat', 0),
                G.nodes[neighbor].get('trans_fat', 0),
                G.nodes[neighbor].get('sodium', 0),
                G.nodes[neighbor].get('total_carbs', 0),
                G.nodes[neighbor].get('fiber', 0),
                G.nodes[neighbor].get('sugars', 0),
                G.nodes[neighbor].get('protein', 0)
            ) for neighbor in neighbors
        ])
        
        if np.std(nutrition_scores) > 0:
            nutrition_norm = (nutrition_scores - nutrition_scores.min()) / (nutrition_scores.max() - nutrition_scores.min())
        else:
            nutrition_norm = np.zeros_like(nutrition_scores)

        
        # Calculate the combined score based on the user-defined alpha
        combined_score = (1 - alpha) * (1 - costs_norm) + alpha * (1 - nutrition_norm)
    
        transition_probs = 1 / (combined_score + 1e-6)  

        transition_probs /= transition_probs.sum()
        
        #Randomly restart (15% change, else weighted walk with probability)
        if random.random() < 0.85: 
            current_node = random.choices(neighbors, weights=transition_probs, k=1)[0]
        else:
            current_node = start_node  
        
        visited_nodes.append(current_node)
    return visited_nodes 


#For comparison against biased
def random_walk(G, start_node, max_steps=100):
    """
    Perform a random walk with equal probability for all neighbors.
    
    G: graph where nodes have nutritional values (calories, protein, fiber, sugars)
       and edges have 'cost'.
    start_node: the starting point for the walk
    max_steps: maximum number of steps in the walk
    """
    current_node = start_node
    visited_nodes = [current_node]

    for step in range(max_steps):
        neighbors = list(G.neighbors(current_node))
        if len(neighbors) == 0:
            break  # No more neighbors, end the walk
        
        # Randomly select the next node
        current_node = random.choice(neighbors)
        visited_nodes.append(current_node)

    return visited_nodes


def calculate_nutritional_score(calories, total_fat, saturated_fat, trans_fat, sodium, total_carbs, fiber, sugars, protein):
    """
    Calculate a nutritional score based on given nutritional values.
    Adjust the weights of each component based on importance.
    """
    return (
        0.1 * calories +      # Weight for calories
        0.1 * protein +       # Weight for protein
        0.1 * fiber +         # Weight for fiber
        0.1 * sugars +        # Weight for sugars
        0.1 * total_fat +     # Weight for total fat
        0.1 * saturated_fat +  # Weight for saturated fat
        0.1 * trans_fat +     # Weight for trans fat
        0.1 * sodium +        # Weight for sodium
        0.1 * total_carbs     # Weight for total carbs
    )

#Hits @ 10 bias walk
def perform_walks(G, start_node, alpha, num_walks=100, max_steps=100):
    all_walks = []
    for i in range(num_walks):
        walk = biased_random_walk_with_weights(G, start_node, alpha, max_steps)
        all_walks.extend(walk)
    
    node_counts = Counter(all_walks)
    del node_counts[start_node]  # Exclude the start node itself

    top_10 = node_counts.most_common(10)
    return top_10



top_10_similar_products_nutrition = perform_walks(G, start_product, alpha=1, num_walks=100, max_steps=50)
top_10_similar_products_cost = perform_walks(G, start_product, alpha=0, num_walks=100, max_steps=50)
top_10_similar_products_middle = perform_walks(G, start_product, alpha=0.5, num_walks=100, max_steps=50)


# Print the top 10 most similar products based on biased random walk
print("Top 10 nutrition products:")
for product, count in top_10_similar_products_nutrition:
    print(f"{product}: visited {count} times")

print('\n')
print("Top 10 cost products cost:")
for product, count in top_10_similar_products_cost:
    print(f"{product}: visited {count} times")

print('\n')
print("Top 10 similar products even:")
for product, count in top_10_similar_products_middle:
    print(f"{product}: visited {count} times")



import matplotlib.pyplot as plt
import numpy as np
import networkx as nx
from sklearn.decomposition import PCA

#Visualizing the data
def visualize_nx_graph_with_embeddings(G):
    # Extract nutritional data for PCA
    nutrient_data = np.array([[G.nodes[node]['calories'], 
                                G.nodes[node]['total_fat'], 
                                G.nodes[node]['saturated_fat'], 
                                G.nodes[node]['trans_fat'], 
                                G.nodes[node]['sodium'], 
                                G.nodes[node]['total_carbs'], 
                                G.nodes[node]['fiber'], 
                                G.nodes[node]['sugars'], 
                                G.nodes[node]['protein']] for node in G.nodes()])

    # Perform PCA to reduce dimensions to 2 for visualization
    pca = PCA(n_components=2)
    pca_result = pca.fit_transform(nutrient_data)

    # Set node positions based on PCA result
    pos = {node: (pca_result[i, 0], pca_result[i, 1]) for i, node in enumerate(G.nodes())}

    # Draw the nodes
    nx.draw_networkx_nodes(G, pos, node_size=50, node_color='lightblue', alpha=0.7)

    # Draw the edges
    nx.draw_networkx_edges(G, pos, width=0.5, alpha=0.5, edge_color='gray')

    # Draw labels with only the name of the node
    labels = {node: node for node in G.nodes()}  # Only show the name of the node
    nx.draw_networkx_labels(G, pos, labels=labels, font_size=8)

    # Set plot title
    plt.title("Nutrition Products Network Graph with PCA")
    plt.xlabel("PCA Component 1")
    plt.ylabel("PCA Component 2")
    plt.axis('off')  # Turn off the axis

    plt.show()

# Call the visualization function
visualize_nx_graph_with_embeddings(G)

#Further see cheapest products
sorted_products = sorted(products, key=lambda x: float(x['Price'].replace('$', '').strip()))

# Print the sorted products
for product in sorted_products:
    print(f"Product: {product['Product Name']}, Price Per Serving: {product['Price']}")


import os
import json

# Get the current directory
current_directory = os.getcwd()

# Dictionary to hold the count of nodes for each JSON file
node_counts = {}

# Iterate through all files in the current directory
for filename in os.listdir(current_directory):
    if filename.endswith('.json'):
        #Construct the full file path
        file_path = os.path.join(current_directory, filename)
        
        # Open the JSON file and load the data
        with open(file_path, 'r') as file:
            
            if isinstance(data, list):
                node_counts[filename] = len(data)  # Count nodes in the list
            elif isinstance(data, dict):  # If data is a dictionary
                node_counts[filename] = len(data.get('nodes', []))  # Adjust based on your JSON structure

# Print the number of nodes for each JSON file
for filename, count in node_counts.items():
    print(f"{filename}: {count} nodes")
