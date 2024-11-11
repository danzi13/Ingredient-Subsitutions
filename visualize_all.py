import json
import re
import pandas as pd
import numpy as np
import networkx as nx
import plotly.graph_objs as go
from sklearn.decomposition import PCA
import os
import plotly.express as px
import plotly.graph_objs as go


# Constants
FL_OZ_TO_ML = 29.5735  # 1 fl oz = 29.5735 ml
STANDARD_SERVING_SIZE_ML = 240  # Standard serving size (240 mL or 1 cup)

def extract_serving_container(servings_text):
    match = re.search(r'(\d+(\.\d+)?)\s?servings', servings_text.lower())
    if match:
        return float(match.group(1))
    return 1.0  

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

# Function to extract and normalize serving size based on fl oz, oz, or cups
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

color_map = {}

def visualize_combined_graph(directory):
    colors = [
        'red', 'green', 'blue', 'yellow', 'orange', 'purple', 
        'cyan', 'magenta', 'brown', 'pink', 'grey', 'black', 
        'violet', 'indigo', 'gold', 'silver', 'teal', 'navy', 
        'coral', 'salmon', 'lightblue', 'lightgreen', 'lavender', 
        'beige', 'turquoise', 'chocolate', 'plum', 'crimson', 
        'olive', 'khaki', 'maroon', 'tan', 'peach'
    ]    
    color_index = -1  # Start with the first color

    G_combined = nx.Graph()
    food_price = []

    # Process each JSON file
    for filename in os.listdir(directory):
        if filename.endswith('.json'):
            file_path = os.path.join(directory, filename)

            # Assign the color for the current JSON file
            current_color = colors[color_index % len(colors)]
            color_map[current_color] = filename  # Map color to the current filename
            color_index += 1

            with open(file_path, 'r') as file:
                products = json.load(file)

                # Assume each product has nutritional data and add it to the graph
                for product in products:
                    product_name = product["Product Name"]

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

                    price = float(product['Price'].replace('$', '').strip())  
                    serving_size = extract_serving_container(serving_size_text) 
                    price_per_container = price / serving_size
                    

                    # Add node to the combined graph
                    G_combined.add_node(product_name,
                                        calories=calories * scaling_factor,
                                        total_fat=total_fat * scaling_factor,
                                        saturated_fat=saturated_fat * scaling_factor,
                                        trans_fat=trans_fat * scaling_factor,
                                        sodium=sodium * scaling_factor,
                                        total_carbs=total_carbs * scaling_factor,
                                        fiber=fiber * scaling_factor,
                                        sugars=sugars * scaling_factor,
                                        protein=protein * scaling_factor,
                                        color=current_color,
                                        price = price_per_container)

                    

                    

    # Perform PCA on the combined graph
    nutrient_data = np.array([[G_combined.nodes[node]['calories'], 
                                G_combined.nodes[node]['total_fat'], 
                                G_combined.nodes[node]['saturated_fat'], 
                                G_combined.nodes[node]['trans_fat'], 
                                G_combined.nodes[node]['sodium'], 
                                G_combined.nodes[node]['total_carbs'], 
                                G_combined.nodes[node]['fiber'], 
                                G_combined.nodes[node]['sugars'], 
                                G_combined.nodes[node]['protein']] 
                               for node in G_combined.nodes()])

    food_price = np.array([G_combined.nodes[node]['price'] for node in G_combined.nodes()])

    # Perform PCA to reduce dimensions to 2 for visualization
    pca = PCA(n_components=1)
    pca_result = pca.fit_transform(nutrient_data)

    #print(food_price.describe())
    #print(pca_result.describe())
    # Create a DataFrame for Plotly Express
    df = pd.DataFrame(pca_result, columns=['score'])

    df['Price'] = food_price
    

    df['Node'] = list(G_combined.nodes())
    df['Color'] = [G_combined.nodes[node]['color'] for node in G_combined.nodes()]

    # Get the corresponding file names for each node
    df['File Name'] = [color_map[G_combined.nodes[node]['color']] for node in G_combined.nodes()]

    # Create the scatter plot using Plotly Express, using 'File Name' for the color key
    fig = px.scatter(df, x='score', y='Price', color='File Name',
                    hover_name='Node', title="Combined Nutrition Products Network Graph with PCA")

    # Show the figure
    fig.show()

    # Call the function with the directory containing JSON files
visualize_combined_graph(os.path.join(os.getcwd(), 'food-json'))
