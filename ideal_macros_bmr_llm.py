import numpy as np
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
import random
from collections import Counter


# Function to calculate Basal Metabolic Rate (BMR) using Mifflin-St Jeor Equation
def calculate_bmr(gender, weight_kg, height_cm, age):
    if gender.lower() == "male":
        bmr = 10 * weight_kg + 6.25 * height_cm - 5 * age + 5
    else:
        bmr = 10 * weight_kg + 6.25 * height_cm - 5 * age - 161
    return bmr

# Function to calculate Total Daily Energy Expenditure (TDEE) based on activity level
def calculate_tdee(bmr, activity_level):
    # Activity levels: sedentary, light, moderate, heavy, very heavy
    activity_factors = {
        "sedentary": 1.2,
        "light": 1.375,
        "moderate": 1.55,
        "heavy": 1.725,
        "very_heavy": 1.9,
    }
    return bmr * activity_factors.get(activity_level.lower(), 1.2)  # Default to sedentary

# Function to calculate macronutrient breakdown for given calorie goal
def calculate_macros(total_calories, goal="maintain"):
    # Macronutrient ratios (in percent)
    if goal.lower() == "gain":
        protein_ratio = 0.25
        fat_ratio = 0.3
        carb_ratio = 0.45
    elif goal.lower() == "cut":
        protein_ratio = 0.3
        fat_ratio = 0.35
        carb_ratio = 0.35
    else:  # Maintain goal
        protein_ratio = 0.25
        fat_ratio = 0.3
        carb_ratio = 0.45

    # Calculate grams of each macronutrient
    protein_grams = (total_calories * protein_ratio) / 4  # 1g protein = 4 calories
    fat_grams = (total_calories * fat_ratio) / 9  # 1g fat = 9 calories
    carb_grams = (total_calories * carb_ratio) / 4  # 1g carbohydrate = 4 calories

    return protein_grams, fat_grams, carb_grams

# Main function to get input and calculate results
def calculate_ideal_macros():
    # Get input from user
    gender = input("Enter gender (male/female): ").strip()
    weight_kg = float(input("Enter weight (kg): ").strip())
    height_cm = float(input("Enter height (cm): ").strip())
    age = int(input("Enter age (years): ").strip())
    activity_level = input("Enter activity level (sedentary, light, moderate, heavy, very_heavy): ").strip().lower()
    goal = input("Enter goal (maintain, cut, gain): ").strip().lower()

    # Calculate BMR and TDEE
    bmr = calculate_bmr(gender, weight_kg, height_cm, age)
    tdee = calculate_tdee(bmr, activity_level)

    # Calculate macronutrients
    protein_grams, fat_grams, carb_grams = calculate_macros(tdee, goal)

    # Output the results
    print("\nYour Ideal Macronutrients Breakdown:")
    print(f"Daily Caloric Needs: {tdee:.0f} calories/day")
    print(f"1. Basal Metabolic Rate (BMR): ~{bmr:.0f} calories/day")
    print(f"2. Activity Level: {activity_level.capitalize()}")
    print(f"3. Goal: {goal.capitalize()}")
    print(f"\nMacronutrient Breakdown:")
    print(f"• Protein: {protein_grams:.0f} grams/day")
    print(f"• Fat: {fat_grams:.0f} grams/day")
    print(f"• Carbs: {carb_grams:.0f} grams/day")
    result = {
        "gender": gender.capitalize(),
        "age(years)":age,
        "height(cm)":height_cm,
        "weight(kg)": weight_kg,
        "tdee": tdee,
        "bmr": bmr,
        "activity_level": activity_level.capitalize(),
        "goal": goal.capitalize(),
        "protein_grams": protein_grams
        
        #"fat_grams": fat_grams,
        #"carb_grams": carb_grams
    }

    # Return the result dictionary
    return result

# Example of calling the function and storing the output
macros = calculate_ideal_macros()







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
#print(normalized_prices)



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
               protein=nutrient_data[i][8],
               tprice=product_prices[i], #Protein
               price=normalized_prices[i]) #Price
    











#protein_goal, fat_goal, carbs_goal = macros["protein_grams"], macros["fat_grams"], macros["carb_grams"]
protein_goal = macros["protein_grams"]

#def find_closest_products_by_macros(G, protein_goal, fat_goal, carbs_goal, max_results=10):

def find_closest_products_by_macros(G, protein_goal, max_results=3):
    """
    Find products that best fulfill the user's protein, fat, and carb needs.
    This is done by calculating the absolute difference between the product's macronutrient content 
    and the user's goal, and then selecting the top products based on the smallest differences.
    """
    product_scores = []
    
    for node in G.nodes():
        # Get the macronutrient values for the current product
        protein = G.nodes[node].get('protein', 0)
        price = G.nodes[node].get('tprice', 0)
        #fat = G.nodes[node].get('total_fat', 0)
        #carbs = G.nodes[node].get('total_carbs', 0)

        # Calculate the absolute differences between the product's macros and the user's goals
        protein_diff = abs(protein - protein_goal)
        #fat_diff = abs(fat - fat_goal)
        #carbs_diff = abs(carbs - carbs_goal)
        
        # Calculate a combined score for the product based on these differences (lower is better)
        #total_diff = protein_diff + fat_diff + carbs_diff
        total_diff = protein_diff
        #print(price)

        # Append the product and its total score to the list
       # product_scores.append((node, total_diff, protein, fat, carbs))
        product_scores.append((node, total_diff, protein, price))

    # Sort products by their total difference score and select the top 'max_results' products
    product_scores.sort(key=lambda x: x[1])
    
    # Get the top 'max_results' products
    top_products = product_scores[:max_results]
    
    return top_products

# Find the top products that fulfill the macronutrient needs
#top_matching_products = find_closest_products_by_macros(G, protein_goal, fat_goal, carbs_goal, max_results=10)
top_matching_products = find_closest_products_by_macros(G, protein_goal, max_results=3)

# Display the results
#print("Top products fulfilling your macronutrient needs:")
#for product, score, protein, fat, carbs in top_matching_products:
#    print(f"{product}: Protein = {protein}g, Fat = {fat}g, Carbs = {carbs}g, Total Score = {score}")

text = []


for product, score, protein, price in top_matching_products:
    servingreq = protein_goal/protein
    pricereq = price * servingreq
    #male
    # print(f"{product}: Protein = {protein}g, Serving required = {servingreq}, total price = {pricereq}")
    text.append(f"{product}: Protein = {protein}g, Serving required = {servingreq:.2f}, total price = {pricereq:.2f}")

print("Top products fulfilling your macronutrient needs:")
print("\n".join(text))
info = macros
print(info)
# Updated query for the LLM
query_text_with_details = (
    f"Your are a recommendation system who finds {name_without_extension}(protein) based food products"
    f" based on their protein need using the users personal information, which are : {info} and I, the user, want to understand the reasoning behind these product recommendations . Here are the results:\n\n"
    f"{text}\n\n"
    f"Please provide an explanation for why these specific products were recommended "
    f"in terms of their heir personal information."
)

# Query Meta AI with the updated text
ai=MetaAI()
response = ai.prompt(message=query_text_with_details)

# Clean and display the response
cleaned_response = response['message'].replace("\n", " ")
print(f"Response from LLM: {cleaned_response}")