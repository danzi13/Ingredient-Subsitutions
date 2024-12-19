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
import os
import pdfplumber
from meta_ai_api import MetaAI



# List of available JSON files
file_names = [
    "bread.json",
    "broth.json",
    "butter.json",
    "cheese.json",
    "cream.json",
    "cruciferous_vegetable.json",
    "fruit.json",
    "leafy_green.json",
    "milk.json",
    "pasta.json",
    "processed_meat.json",
    "red_meat.json",
    "rice.json",
    "root_and_bulb_veggies.json",
    "shellfish.json",
    "tortilla.json",
    "yogurt.json",
]

# Corresponding start products for each file
start_products = [
    "Nature's Own Butterbread, Sliced White Bread, 20 oz Loaf",
    "365 by Whole Foods Market, Organic Chicken Broth, 48 Fl Oz",
    "365 by Whole Foods Market, Butter Salted, 16 Ounce",
    "Amazon Brand - Happy Belly Grated Parmesan Cheese Shaker, 16 ounce (Pack of 1)",
    "365 by Whole Foods Market, Heavy Cream Organic, 16 Ounce",
    "365 by Whole Foods Market, Organic Broccoli Florets, 10 Ounce",
    "Banana Bunch",
    "365 by Whole Foods Market, Salad Bag Spinach Baby Organic, 5 Ounce",
    "365 by Whole Foods Market, Milk Whole Organic, 128 Fl Oz",
    "Banza Rotini Pasta from Chickpeas - Gluten Free, High Protein, and Lower Carb Protein Rotini Chickpea Pasta - 8oz",
    "365 by Whole Foods Market, Center Cut Smokehouse Uncured Bacon, Reduced Sodium, 12 oz",
    "Boneless Beef New York Strip Loin Steak, Step 1",
    "Mahatma Indian Basmati Rice, 5lb Bag of Rice, Fluffy, Floral, and Nutty-Flavored Rice, Stovetop or Microwave Rice",
    "Love Beets Organic Cooked Beets, 8.8 oz",
    "Whole Catch, Key West Pink Shrimp 51-60, 12 Ounce (Frozen)",
    "365 by Whole Foods Market Flour Tortillas, 8 ct, 10.7 oz total",
    "Forager Project Cashewmilk Yogurt, Unsweetened Plain, 24-Ounce",
]

# Format file names for user display
formatted_names = [os.path.splitext(name)[0].replace("_", " ").capitalize() for name in file_names]

# Display options to the user
print("Select a type of food product by entering its number:")
for i, name in enumerate(formatted_names, start=1):
    print(f"{i}) {name}")

# Take user input
try:
    choice = int(input("Enter the number corresponding to your choice: ")) - 1
    if 0 <= choice < len(file_names):
        file_name = file_names[choice]
        name_without_extension = os.path.splitext(file_name)[0]
        print(f"You selected: {name_without_extension}")
        
        # Load the JSON file
        with open(file_name, 'r') as file:
            products = json.load(file)
        print("File loaded successfully!")

        # Set the start product based on the user's choice
        start_product = start_products[choice]
        print(f"Start product: {start_product}")
    else:
        print("Invalid selection.")
except ValueError:
    print("Please enter a valid number.")

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


options = {
    "1": ("nutrition", top_10_similar_products_nutrition, "Top nutrition products"),
    "2": ("cost", top_10_similar_products_cost, "Top cost products"),
    "3": ("similar", top_10_similar_products_middle, "Top similar products even"),
}

# Display the options
print("Choose one of the following options:")
print("1) Nutrition")
print("2) Cost")
print("3) Similar")

# Take user input
choice = input("Enter your choice (1, 2, or 3): ")




import matplotlib.pyplot as plt
import numpy as np
import networkx as nx
from sklearn.decomposition import PCA


    
# Call the visualization function


#Further see cheapest products
sorted_products = sorted(products, key=lambda x: float(x['Price'].replace('$', '').strip()))

# Print the sorted products
#for product in sorted_products:
    #print(f"Product: {product['Product Name']}, Price Per Serving: {product['Price']}")

# Format the top N most similar products with additional details
def get_product_details(product_name, products):
    """
    Retrieve details for a product given its name.
    """
    for product in products:
        if product['Product Name'] == product_name:
            #print(product)
            return product
    return {}

# Updated recommendation formatter
def format_recommendations_with_details(top_products, products, option_label):
    """
    Format recommendations with additional details from the JSON file.
    """
    recommendations = []
    for product_name, count in top_products:
        product_details = get_product_details(product_name, products)
        if product_details:
            # Include relevant details (adjust keys based on JSON structure)
            if product_details:
    # Include relevant details
                product_name = product_details.get("Product Name", "N/A")
                price = product_details.get("Price", "N/A")
                serving_size = product_details.get("Servings", "N/A")

                # Extract top-level nutrients
                calories = product_details.get("Calories", "N/A")
                fat_details = product_details.get("Fat", {})
                total_fat = fat_details.get("Total Fat", "N/A")
                saturated_fat = fat_details.get("Saturated Fat", "N/A")
                trans_fat = fat_details.get("Trans Fat", "N/A")

                cholesterol = product_details.get("Cholesterol", "N/A")
                sodium = product_details.get("Sodium", "N/A")

                # Extract carbohydrates and sugars
                carbs_details = product_details.get("Carbs", {})
                total_carbs = carbs_details.get("Total Carbs", "N/A")
                fiber = carbs_details.get("Fiber", "N/A")
                sugars = carbs_details.get("Sugars", "N/A")
                added_sugars = carbs_details.get("Added Sugars", "N/A")

                # Extract protein
                protein = product_details.get("Protein", "N/A")

                # Extract micronutrients
                micronutrients = product_details.get("Micronutrients", {})
                folate = micronutrients.get("Folate", "N/A")
                vitamin_c = micronutrients.get("Vitamin C", "N/A")
                iron = micronutrients.get("Iron", "N/A")
                iodine = micronutrients.get("Iodine", "N/A")
                vitamin_a = micronutrients.get("Vitamin A", "N/A")
                zinc = micronutrients.get("Zinc", "N/A")
                calcium = micronutrients.get("Calcium", "N/A")
                potassium = micronutrients.get("Potassium", "N/A")
                vitamin_d = micronutrients.get("Vitamin D", "N/A")

                # Extract ingredients
                ingredients = product_details.get("Ingredients", "N/A")


            # Add formatted details to the recommendations
            recommendations.append(
                f"{product_name}: visited {count} times\n"
                f"  Price: {price}\n"
                f"  Serving Size: {serving_size}\n"
                f"  Calories: {calories}\n"
                f"  Fat Details:\n"
                f"    Total Fat: {total_fat}, Saturated Fat: {saturated_fat}, Trans Fat: {trans_fat}\n"
                f"  Cholesterol: {cholesterol}\n"
                f"  Sodium: {sodium}\n"
                f"  Carbohydrate Details:\n"
                f"    Total Carbs: {total_carbs}, Fiber: {fiber}, Sugars: {sugars}, Added Sugars: {added_sugars}\n"
                f"  Protein: {protein}\n"
                f"  Micronutrients:\n"
                f"    Folate: {folate}, Vitamin C: {vitamin_c}, Iron: {iron}, Iodine: {iodine},\n"
                f"    Vitamin A: {vitamin_a}, Zinc: {zinc}, Calcium: {calcium},\n"
                f"    Potassium: {potassium}, Vitamin D: {vitamin_d}\n"
                f"  Ingredients: {ingredients}\n"
            )

    return f"{option_label}:\n" + "\n\n".join(recommendations)

# Format recommendations for each option
top_nutrition_recommendations = format_recommendations_with_details(
    top_10_similar_products_nutrition, products, "Top nutrition products"
)
top_cost_recommendations = format_recommendations_with_details(
    top_10_similar_products_cost, products, "Top cost products"
)
top_middle_recommendations = format_recommendations_with_details(
    top_10_similar_products_middle, products, "Top similar products even"
)

# Select and display the user's choice
if choice in options:
    option_name, top_products, option_label = options[choice]
    selected_recommendations = format_recommendations_with_details(
        top_products, products, option_label
    )
    print("\nFormatted Recommendation with Details:")
    print(selected_recommendations)
else:
    print("Invalid choice. Please choose 1, 2, or 3.")

# Updated query for the LLM
query_text_with_details = (
    f"Your are a recommendation system who finds {name_without_extension} based food products based on user preferences for {option_name} quality using and I, the user, want to understand the reasoning behind these product recommendations "
    f"in terms of nutrition, cost(per serving), and an even balance. Here are the results:\n\n"
    f"{selected_recommendations}\n\n"
    f"Please provide an explanation for why these specific products were recommended "
    f"in terms of nutrition, cost, and an even balance."
)

# Query Meta AI with the updated text
ai=MetaAI()
response = ai.prompt(message=query_text_with_details)

# Clean and display the response
cleaned_response = response['message'].replace("\n", " ")
print(f"Response from LLM: {cleaned_response}")









