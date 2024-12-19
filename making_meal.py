import os
import json
import re
import numpy as np
import networkx as nx
import random
import itertools


# Constants
valid_combinations = []
loss_threshold = 2000


FL_OZ_TO_ML = 29.5735
STANDARD_SERVING_SIZE_ML = 240  # Standard serving size (240 mL or 1 cup)

# Ideal macros with sodium in grams for uniformity

ideal_macros = {
    'calories': 700,       
    'total_fat': 23,       
    'saturated_fat': 7,    
    'trans_fat': 0,       
    'sodium': 1.0,         
    'total_carbs': 80,    
    'fiber': 8,           
    'sugars': 12,          
    'protein': 35         
}



# Helper Functions
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
        return float(re.sub(r'[^\d.]', '', value)) / 1000  # Convert mg to grams
    elif 'mcg' in value:
        return float(re.sub(r'[^\d.]', '', value)) / 1e6   # Convert mcg to grams
    else:
        return float(re.sub(r'[^\d.]', '', value))  # Already in grams

def get_serving_size(serving_size_text):
    """
    Extracts the serving size in grams from the text.
    Handles cases for explicit grams, pieces, ounces, and cups.
    """
    serving_size_text = serving_size_text.lower()
    OZ_TO_GRAMS = 28.3495  # Conversion factor from oz to grams
    TBSP_TO_GRAMS = 15  # Approximate weight for 1 tbsp
    MUFFIN_GRAMS = 240  # Default weight for one muffin in grams

    # Match explicit grams
    match_grams = re.search(r'(\d+)\s?g', serving_size_text)
    if match_grams:
        return float(match_grams.group(1))

    # Match ounces
    match_oz = re.search(r'(\d+(\.\d+)?)\s?oz', serving_size_text)
    if match_oz:
        return float(match_oz.group(1)) * OZ_TO_GRAMS

    # Match tablespoons
    match_tbsp = re.search(r'(\d+(\.\d+)?)\s?tbsp', serving_size_text)
    if match_tbsp:
        return float(match_tbsp.group(1)) * TBSP_TO_GRAMS
    
    match_pieces = re.search(r'(\d+)\s?pieces?', serving_size_text)
    if match_pieces:
        piece_count = float(match_pieces.group(1))
        piece_weight_grams = PIECES_TO_OZ * OZ_TO_GRAMS
        return piece_count * piece_weight_grams


    # Match muffins
    match_muffins = re.search(r'(\d+(\.\d+)?)\s?muffins?', serving_size_text)
    if match_muffins:
        return float(match_muffins.group(1)) * MUFFIN_GRAMS

    # Match cups
    match_cups = re.search(r'(\d+(\.\d+)?|\d+\s+1/2|\d+)\s?cups?', serving_size_text)
    if match_cups:
        cup_value = eval(match_cups.group(1).replace(" ", "+"))  # Evaluate fractions
        return cup_value * STANDARD_SERVING_SIZE_ML

    raise ValueError(f"Could not convert serving size to grams: {serving_size_text}")

def normalize_nutrients_by_serving(product, gram_weight):
    """
    Normalize nutrients based on actual serving size in grams.
    """
    nutrients = {
        'calories': float(product["Calories"]) if product["Calories"] else 0.0,
        'total_fat': convert_to_grams(product["Fat"]["Total Fat"]),
        'saturated_fat': convert_to_grams(product["Fat"]["Saturated Fat"]),
        'trans_fat': convert_to_grams(product["Fat"]["Trans Fat"]),
        'sodium': convert_to_grams(product["Sodium"]),
        'total_carbs': convert_to_grams(product["Carbs"]["Total Carbs"]),
        'fiber': convert_to_grams(product["Carbs"]["Fiber"]),
        'sugars': convert_to_grams(product["Carbs"]["Sugars"]),
        'protein': convert_to_grams(product["Protein"]),
    }
    return {key: max(0, value * (gram_weight / STANDARD_SERVING_SIZE_ML)) for key, value in nutrients.items()}

def process_product(product):
    """
    Process each product to extract relevant data and validate nutrient values.
    """
    try:
        serving_size_text = product["Servings"]
        gram_weight = get_serving_size(serving_size_text)
        nutrients = normalize_nutrients_by_serving(product, gram_weight)
        price = float(product['Price'].replace('$', '').strip())
        servings_per_container = extract_serving_container(serving_size_text)
        price_per_serving = price / servings_per_container if servings_per_container > 0 else price
        return {
            "name": product["Product Name"],
            **nutrients,
            "gram_weight": gram_weight,
            "price_per_serving": price_per_serving
        }
    except Exception as e:
        #print(f"Error processing product: {product.get('Product Name', 'Unknown')} - {e}")
        return None  # Skip invalid entries

def build_graphs(categories, food_json_dir):
    """
    Build graphs from JSON files, with error-checking for invalid data.
    """
    graphs = {}
    for category in categories:
        graph = nx.Graph()
        file_path = os.path.join(food_json_dir, category)
        with open(file_path, 'r') as file:
            products = json.load(file)
            for product in products:
                processed_product = process_product(product)
                if processed_product:  # Only add valid products
                    graph.add_node(
                        processed_product["name"],
                        **{key: processed_product[key] for key in ideal_macros.keys()},
                        gram_weight=processed_product["gram_weight"],
                        price=processed_product["price_per_serving"]
                    )
        graphs[category] = graph
    return graphs

def calculate_combined_nutrition(selected_nodes, graphs, scaling_factors):
    """
    Calculate combined nutrition for the selected nodes with scaling factors applied.
    """
    combined_nutrition = {key: 0 for key in ideal_macros.keys()}
    for graph_key, node in selected_nodes.items():
        node_data = graphs[graph_key].nodes[node]
        scale = scaling_factors[graph_key]
        for key in ideal_macros.keys():
            combined_nutrition[key] += max(0, node_data[key] * scale)
    return combined_nutrition

def calculate_loss(nutrition, ideal_macros):
    """
    Calculate the squared loss from ideal macros.
    """
    weights = {
        'calories': 1.0,  # Equal weighting
        'total_fat': 1.0,
        'saturated_fat': 1.0,
        'trans_fat': 1.0,
        'sodium': 1.0,
        'total_carbs': 1.0,
        'fiber': 1.0,
        'sugars': 1.0,
        'protein': 1.0
    }
    return sum(weights[key] * (nutrition[key] - ideal_macros[key])**2 for key in ideal_macros)
    return sum((nutrition[key] - ideal_macros[key])**2 for key in ideal_macros)



def iterative_optimization(graphs, ideal_macros, iterations):
    """
    Optimize meal selection by refining choices iteratively with error-checking,
    exploring configurations by adjusting scales even if the configuration was visited.
    Collects all valid combinations within a loss threshold.
    """
    global_best_loss = float('inf')  # Keep track of the best loss across all iterations
    global_best_combination = None  # Track the best combination across all iterations
    global_best_scaling_factors = None

    # Initialize random selections and scaling factors
    selected_nodes = {key: random.choice(list(graph.nodes)) for key, graph in graphs.items()}
    scaling_factors = {key: 1.0 for key in graphs.keys()}
    best_nutrition = calculate_combined_nutrition(selected_nodes, graphs, scaling_factors)
    best_loss = calculate_loss(best_nutrition, ideal_macros)

    # Update global best if this initialization is better
    if best_loss < global_best_loss:
        global_best_loss = best_loss
        global_best_combination = selected_nodes.copy()
        global_best_scaling_factors = scaling_factors.copy()

    for iteration in range(iterations):
        if iteration % 1000 == 0:
            print(f"Iteration {iteration}: Best Loss = {global_best_loss:.4f}")

        # Restart every 250 iterations with new random selections
        if iteration % 250 == 0:
            selected_nodes = {key: random.choice(list(graph.nodes)) for key, graph in graphs.items()}
            scaling_factors = {key: 1.0 for key in graphs.keys()}
            best_nutrition = calculate_combined_nutrition(selected_nodes, graphs, scaling_factors)
            best_loss = calculate_loss(best_nutrition, ideal_macros)

        for graph_key, graph in graphs.items():
            current_node = selected_nodes[graph_key]
            best_node = current_node
            best_scale = scaling_factors[graph_key]
            best_trial_loss = best_loss

            # Explore neighbors and randomly pick a scale
            for neighbor in graph.nodes:
                scale = random.choice([0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0])  # Random scale
                trial_nodes = selected_nodes.copy()
                trial_factors = scaling_factors.copy()
                trial_nodes[graph_key] = neighbor
                trial_factors[graph_key] = scale

                # Calculate trial nutrition and loss
                trial_nutrition = calculate_combined_nutrition(trial_nodes, graphs, trial_factors)
                trial_loss = calculate_loss(trial_nutrition, ideal_macros)

                # Add to valid combinations if within threshold
                combination_key = tuple(trial_nodes.values())
                existing_combination = next(
                    (c for c in valid_combinations if tuple(c["combination"].values()) == combination_key), None
                )

                if trial_loss <= loss_threshold:
                    if existing_combination:
                        # Replace only if the new loss is better
                        if trial_loss < existing_combination["loss"]:
                            existing_combination["scaling_factors"] = trial_factors.copy()
                            existing_combination["nutrition"] = trial_nutrition.copy()
                            existing_combination["loss"] = trial_loss
                    else:
                        # Add new valid combination
                        valid_combinations.append({
                            "combination": trial_nodes.copy(),
                            "scaling_factors": trial_factors.copy(),
                            "nutrition": trial_nutrition.copy(),
                            "loss": trial_loss
                        })

                # Update best node and scale if trial loss improves
                if trial_loss < best_trial_loss:
                    best_trial_loss = trial_loss
                    best_node = neighbor
                    best_scale = scale

            # Update the selected node and scaling factor for this graph
            selected_nodes[graph_key] = best_node
            scaling_factors[graph_key] = best_scale

            # Update the best nutrition and loss
            best_nutrition = calculate_combined_nutrition(selected_nodes, graphs, scaling_factors)
            best_loss = calculate_loss(best_nutrition, ideal_macros)

            # Update global best if this trial is better
            if best_loss < global_best_loss:
                global_best_loss = best_loss
                global_best_combination = selected_nodes.copy()
                global_best_scaling_factors = scaling_factors.copy()

    return global_best_combination, global_best_scaling_factors, calculate_combined_nutrition(global_best_combination, graphs, global_best_scaling_factors)

# Grouping of food categories for each meal
meal_combinations = {
    #"Breakfast": [
    #    ["bread.json", "butter.json", "fruit.json", "yogurt.json"],  # Bread + Butter + Fruit + Yogurt
    #    ["tortilla.json", "cheese.json", "milk.json"],  # Tortilla + Cheese + Milk
    #    ["bread.json", "cream.json", "fruit.json"],  # Bread + Cream + Fruit
    #    ["leafy_green.json", "cheese.json", "red_meat.json"],  # Leafy Greens + Cheese + Red Meat (Breakfast Omelet)
    #    ["yogurt.json", "fruit.json", "root_and_bulb_veggies.json"],  # Yogurt + Fruit + Root Vegetables
    #],
    #"Lunch": [
    #    ["bread.json", "cheese.json", "leafy_green.json", "processed_meat.json"],  # Sandwich
    #    ["rice.json", "shellfish.json", "cruciferous_vegetable.json"],  # Seafood Stir-Fry
    #    ["tortilla.json", "cheese.json", "red_meat.json", "leafy_green.json"],  # Wrap
    #    ["pasta.json", "cream.json", "processed_meat.json", "root_and_bulb_veggies.json"],  # Pasta Dish
    #    ["broth.json", "root_and_bulb_veggies.json", "cruciferous_vegetable.json", "red_meat.json"],  # Hearty Soup
    #],

    "Dinner": [
        #["rice.json", "red_meat.json", "cruciferous_vegetable.json"], 
        #["pasta.json", "cheese.json", "leafy_green.json", "processed_meat.json"],  
        #["broth.json", "shellfish.json", "root_and_bulb_veggies.json", "cruciferous_vegetable.json"],  
        #["tortilla.json", "cheese.json", "processed_meat.json", "leafy_green.json"], 
        ["bread.json", "cream.json", "red_meat.json", "cruciferous_vegetable.json"] 
        #["broth.json", "root_and_bulb_veggies.json", "cruciferous_vegetable.json", "red_meat.json"]
    ]

}

#meal_combinations = {}


# Main Execution
def main():
    categories = ["pasta.json", "cheese.json", "leafy_green.json", "processed_meat.json"]
    food_json_dir = 'food-JSON'
    graphs = build_graphs(categories, food_json_dir)

    iterations = 10000
    selected_nodes, scaling_factors, final_nutrition = iterative_optimization(graphs, ideal_macros, iterations)

    print("\nBest configuration after all iterations:")
    for graph_key, node in selected_nodes.items():
        scale = scaling_factors[graph_key]
        print(f"Graph: {graph_key}, Node: {node}, Scaling Factor: {scale:.2f}")

    print("\nFinal combined nutrition:")
    for key, value in final_nutrition.items():
        print(f"{key}: {value:.2f}")

    print("\nDifference from ideal macros:")
    for key in ideal_macros:
        difference = final_nutrition[key] - ideal_macros[key]
        print(f"{key}: {difference:.2f}")

    print("\nFinal Loss:")
    print(f"Loss: {calculate_loss(final_nutrition, ideal_macros):.4f}")


    print(len(valid_combinations))

    #for i, item in enumerate(valid_combinations):
    #    print(item)

    # Global variable to store valid combinations
    global_best_loss = float('inf')  # Keep track of the best loss across all iterations
    global_best_combination = None  # Track the best combination across all iterations
    global_best_scaling_factors = None  # Track scaling factors of the best combination

    print("Starting meal combination optimization...")

    # Iterate through each meal type and combination
    for meal_type, combinations in meal_combinations.items():
        print(f"\nProcessing {meal_type} combinations...")

        for combo in combinations:
            print(f"\nEvaluating combination: {combo}")

            # Build graphs for the current combination
            categories = combo
            food_json_dir = 'food-JSON'
            graphs = build_graphs(categories, food_json_dir)

            # Run optimization
            iterations = 10000
            selected_nodes, scaling_factors, final_nutrition = iterative_optimization(graphs, ideal_macros, iterations)

            # Print the best configuration
            print("\nBest configuration after all iterations:")
            for graph_key, node in selected_nodes.items():
                scale = scaling_factors[graph_key]
                print(f"Graph: {graph_key}, Node: {node}, Scaling Factor: {scale:.2f}")

            # Print the final combined nutrition
            print("\nFinal combined nutrition:")
            for key, value in final_nutrition.items():
                print(f"{key}: {value:.2f}")

            # Print the difference from ideal macros
            print("\nDifference from ideal macros:")
            for key in ideal_macros:
                difference = final_nutrition[key] - ideal_macros[key]
                print(f"{key}: {difference:.2f}")

            # Print the final loss
            print("\nFinal Loss:")
            print(f"Loss: {calculate_loss(final_nutrition, ideal_macros):.4f}")

    print(len(valid_combinations))

    # Sort valid combinations based on nutrition, cost-nutrition ratio, and cost
    top_nutrition = sorted(valid_combinations, key=lambda x: x["loss"])[:2]

    import os
    import json

    food_json_dir = "food-JSON"  # Directory containing the JSON files



    def find_price_in_json(node_name, graph_key):
        """Search for the price of a node in the JSON file corresponding to the graph_key."""
        file_path = os.path.join(food_json_dir, graph_key)
        try:
            with open(file_path, 'r') as file:
                data = json.load(file)
                for product in data:
                    if product["Product Name"] == node_name:
                        return float(product["Price"].replace("$", "").strip())
        except Exception as e:
            print(f"Error reading {file_path}: {e}")
        return None

    # Calculate the total cost for each combination in valid_combinations
    for combo in valid_combinations:
        total_cost = 0
        for graph_key, scaling_factor in combo["scaling_factors"].items():
            node_name = combo["combination"][graph_key]

            # Try retrieving from the graph first
            if graph_key in graphs and node_name in graphs[graph_key].nodes:
                node_data = graphs[graph_key].nodes[node_name]
                price_per_serving = node_data["price"]
                

            else:
                # Search for the price in the corresponding JSON if missing in the graph
                price_per_serving = find_price_in_json(node_name, graph_key)
                if price_per_serving is None:
                    print(f"Warning: Price for node '{node_name}' in graph '{graph_key}' not found. Skipping.")
                    continue

            # Calculate total cost
            total_cost += price_per_serving * scaling_factor

        combo["total_cost"] = total_cost  # Store the total cost in the combination dictionary

    # Sort by actual total cost

    top_cost = sorted(valid_combinations, key=lambda x: x["total_cost"])[:2]
    top_cost_nutrition = sorted(
        valid_combinations,
        key=lambda x: x["loss"] * x["total_cost"]
    )[:2]


    # Display results
    # Top 5 by Nutrition
    print("\nTop 5 by Nutrition:")
    for i, combo in enumerate(top_nutrition, 1):
        print(f"Rank {i}: Loss = {combo['loss']:.4f}")
        print(f"  Total Cost: {combo['total_cost']:.2f}")
        print("  Combination:")
        for graph_key, node in combo['combination'].items():
            scale = combo['scaling_factors'][graph_key]
            print(f"    Graph: {graph_key}, Node: {node}")
        print("\n  Nutrition:")
        for key, value in combo['nutrition'].items():
            print(f"    {key}: {value:.2f}")
        print("-" * 50)

    # Top 5 by Cost-Nutrition Ratio
    print("\nTop 5 by Cost-Nutrition Ratio:")
    for i, combo in enumerate(top_cost_nutrition, 1):
        print(f"\nRank {i}: Combined Score (Loss * Cost) = {combo['loss'] * combo['total_cost']:.4f}")
        print(f"  Total Cost: {combo['total_cost']:.2f}")
        print(f"  Loss: {combo['loss']:.4f}")
        print("  Combination:")
        for graph_key, node in combo['combination'].items():
            scale = combo['scaling_factors'][graph_key]
            print(f"    Graph: {graph_key}, Node: {node}, Scaling Factor: {scale:.2f}")

        print("\n  Nutrition:")
        for key, value in combo['nutrition'].items():
            print(f"    {key}: {value:.2f}")
        print("-" * 50)

    # Top 5 by Cost
    print("\nTop 5 by Cost:")
    for i, combo in enumerate(top_cost, 1):
        print(f"Rank {i}: Total Cost = {combo['total_cost']:.2f}")
        print(f"  Loss: {combo['loss']:.4f}")

        print("  Combination:")
        for graph_key, node in combo['combination'].items():
            scale = combo['scaling_factors'][graph_key]
            print(f"    Graph: {graph_key}, Node: {node}, Scaling Factor: {scale:.2f}")
        
        print("\n  Nutrition:")
        for key, value in combo['nutrition'].items():
            print(f"    {key}: {value:.2f}")
        print("-" * 50)



    print(len(valid_combinations))


    import os
    import json

    # Directory containing the JSON files


    # Loop through JSON files and process
    sorted_results = {}

    # Create a global list to store all nodes with their price per serving
    all_nodes = []

    # Extract nodes and prices from all graphs
    for category, graph in graphs.items():
        for node, data in graph.nodes(data=True):
            price = data.get("price", 0.0)
            all_nodes.append((node, price, category))  # Include category for context

    # Sort all nodes by price per serving in descending order
    sorted_nodes = sorted(all_nodes, key=lambda x: x[1], reverse=True)

    # Print sorted nodes
    #print("\nAll Nodes Sorted by Price per Serving (Most Expensive First):")
    #for node, price, category in sorted_nodes:
    #    print(f"  Node: {node}, Category: {category}, Price per Serving: ${price:.2f}")


    
if __name__ == "__main__":
    main()