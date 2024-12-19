import matplotlib.pyplot as plt
import numpy as np
import random
from making_meal import build_graphs, calculate_combined_nutrition, calculate_loss, ideal_macros, meal_combinations

# [4243.825, 7037.525, 10856.3, 17156.375, 22668.775]

#NEW: '''3

#Total Time for 25000 iterations: 4402.43 seconds
4402 / 25000 * 40
#Average Time Per Iteration: 0.176097 seconds
#[4243.825, 7037.525, 10856.3, 17156.375, 22668.775]

#Total Time for 10000 iterations: 4357.13 seconds //// 275.410648753 per second
'''
Top 5 by Nutrition:
Rank 1: Loss = 22.0434
  Total Cost: 20.90
  Combination:
    Graph: bread.json, Node: 365 By Whole Foods Market, Frozen, Everything Bagels 6 Count, 18 Ounce
    Graph: cream.json, Node: Double Devon Cream (6 ounce)
    Graph: red_meat.json, Node: DUBRETON Organic Ground Pork, 16 OZ
    Graph: cruciferous_vegetable.json, Node: Taylor Farms Shredded Red Cabbage, 8 oz

  Nutrition:
    calories: 699.25
    total_fat: 21.28
    saturated_fat: 9.57
    trans_fat: 1.42
    sodium: 0.91
    total_carbs: 77.28
    fiber: 7.83
    sugars: 9.83
    protein: 34.62
--------------------------------------------------
Rank 2: Loss = 28.5738
  Total Cost: 64.04
  Combination:
    Graph: bread.json, Node: Dave's Killer Bread Killer Classic English Muffins, Organic English Muffins, 6 Count
    Graph: cream.json, Node: Devon Cream Company Clotted Cream, 6 oz
    Graph: red_meat.json, Node: Ground Lamb, 1lb
    Graph: cruciferous_vegetable.json, Node: Taylor Farms Shredded Red Cabbage, 8 oz

  Nutrition:
    calories: 698.08
    total_fat: 22.10
    saturated_fat: 10.03
    trans_fat: 1.19
    sodium: 0.90
    total_carbs: 77.16
    fiber: 7.83
    sugars: 9.71
    protein: 34.62
--------------------------------------------------

Top 5 by Cost-Nutrition Ratio:

Rank 1: Combined Score (Loss * Cost) = 564.3636
  Total Cost: 5.41
  Loss: 104.3234
  Combination:
    Graph: rice.json, Node: Lotus Foods Bulk Organic Forbidden Rice - Black Rice Organic, Purple Rice, Black Rice Bulk, Gluten Free Heirloom Rice, Whole Grain, Non GMO, Vegan - 11 lb Bag, Scaling Factor: 2.50
    Graph: red_meat.json, Node: 365 By Whole Foods Market, Beef Ground 80% Lean/20% Fat, 16 Ounce, Scaling Factor: 2.50
    Graph: cruciferous_vegetable.json, Node: 365 by Whole Foods Market, Root Vegetables Organic, 16 Ounce, Scaling Factor: 2.00

  Nutrition:
    calories: 702.29
    total_fat: 28.79
    saturated_fat: 9.33
    trans_fat: 1.75
    sodium: 0.12
    total_carbs: 81.75
    fiber: 6.29
    sugars: 4.92
    protein: 34.71
--------------------------------------------------

Rank 2: Combined Score (Loss * Cost) = 669.9496
  Total Cost: 7.58
  Loss: 88.3868
  Combination:
    Graph: rice.json, Node: Lotus Foods Bulk Organic Forbidden Rice - Black Rice Organic, Purple Rice, Black Rice Bulk, Gluten Free Heirloom Rice, Whole Grain, Non GMO, Vegan - 11 lb Bag, Scaling Factor: 2.50
    Graph: red_meat.json, Node: 365 By Whole Foods Market, Beef Ground 80% Lean/20% Fat, 16 Ounce, Scaling Factor: 2.50
    Graph: cruciferous_vegetable.json, Node: Taylor Farms Shredded Red Cabbage, 8 oz, Scaling Factor: 3.50

  Nutrition:
    calories: 701.41
    total_fat: 28.79
    saturated_fat: 9.33
    trans_fat: 1.75
    sodium: 0.12
    total_carbs: 81.40
    fiber: 6.65
    sugars: 5.80
    protein: 33.82
--------------------------------------------------

Top 5 by Cost:
Rank 1: Total Cost = 4.34
  Loss: 943.6026
  Combination:
    Graph: rice.json, Node: Lotus Foods Bulk Organic Forbidden Rice - Black Rice Organic, Purple Rice, Black Rice Bulk, Gluten Free Heirloom Rice, Whole Grain, Non GMO, Vegan - 11 lb Bag, Scaling Factor: 3.50
    Graph: red_meat.json, Node: 365 by Whole Foods Market 365 Hot Italian Pork Sausage, 16 Ounce, Scaling Factor: 3.50
    Graph: cruciferous_vegetable.json, Node: Organic Cauliflower, 1 Each, Scaling Factor: 0.50

  Nutrition:
    calories: 688.67
    total_fat: 19.73
    saturated_fat: 5.12
    trans_fat: 0.00
    sodium: 0.66
    total_carbs: 105.24
    fiber: 6.11
    sugars: 4.18
    protein: 25.04
--------------------------------------------------
Rank 2: Total Cost = 4.71
  Loss: 379.5870
  Combination:
    Graph: rice.json, Node: Lotus Foods Bulk Organic Forbidden Rice - Black Rice Organic, Purple Rice, Black Rice Bulk, Gluten Free Heirloom Rice, Whole Grain, Non GMO, Vegan - 11 lb Bag, Scaling Factor: 2.50
    Graph: red_meat.json, Node: Beef Chuck Short Rib Bone-In Step 1, Scaling Factor: 4.50
    Graph: cruciferous_vegetable.json, Node: 365 by Whole Foods Market, Root Vegetables Organic, 16 Ounce, Scaling Factor: 1.50

  Nutrition:
    calories: 692.97
    total_fat: 22.26
    saturated_fat: 5.26
    trans_fat: 1.21
    sodium: 0.12
    total_carbs: 79.80
    fiber: 5.76
    sugars: 4.21
    protein: 51.08
--------------------------------------------------
9917
(base) danzis-air:amazon_scraper michaeldanzi$ '''

# Function to generate varied macro targets
def generate_random_macros(base_macros, variation=0.1, n=20):
    """Generate N macro targets by adding Gaussian noise."""
    random_macros = []
    for _ in range(n):
        noisy_macros = {k: max(0, v + np.random.normal(0, variation * v)) for k, v in base_macros.items()}
        random_macros.append(noisy_macros)
    return random_macros

# Optimized iterative function with loss tracking
def iterative_optimization_with_tracking(graphs, ideal_macros, iterations, loss_thresholds):
    """
    Optimize meal selection and track iterations that hit loss thresholds.
    Fully explores neighbors and scales, replacing global tracking with loss threshold tracking.
    """
    # Initialize best loss and selections
    best_loss = float('inf')
    best_combination = None
    best_scaling_factors = None
    
    # Loss thresholds tracking
    loss_hits = {threshold: None for threshold in loss_thresholds}

    # Initialize random selections and scaling factors
    selected_nodes = {key: random.choice(list(graph.nodes)) for key, graph in graphs.items()}
    scaling_factors = {key: 1.0 for key in graphs.keys()}
    best_nutrition = calculate_combined_nutrition(selected_nodes, graphs, scaling_factors)
    best_loss = calculate_loss(best_nutrition, ideal_macros)

    for iteration in range(iterations):

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
                for scale in [0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0]:  # Test realistic scaling factors
                    trial_nodes = selected_nodes.copy()
                    trial_factors = scaling_factors.copy()
                    trial_nodes[graph_key] = neighbor
                    trial_factors[graph_key] = scale

                    # Calculate trial nutrition and loss
                    trial_nutrition = calculate_combined_nutrition(trial_nodes, graphs, trial_factors)
                    trial_loss = calculate_loss(trial_nutrition, ideal_macros)

                    # Track the first iteration that hits each loss threshold
                    for threshold in loss_thresholds:
                        if trial_loss <= threshold and loss_hits[threshold] is None:
                            loss_hits[threshold] = iteration

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

    return loss_hits, best_loss


# Measure effectiveness across multiple macro sets and meal combinations
def measure_effectiveness(meal_combos, ideal_macros_list, iterations, loss_thresholds):
    """Run optimization for multiple macros and meals, track thresholds."""
    results = {threshold: [] for threshold in loss_thresholds}

    for macros_idx, macros in enumerate(ideal_macros_list):
        print(f"\nRunning Macro Set {macros_idx + 1}/{len(ideal_macros_list)}...")
        
        for meal in meal_combos:
            graphs = build_graphs(meal, 'food-JSON')
            loss_hits, _ = iterative_optimization_with_tracking(graphs, macros, iterations, loss_thresholds)
            
            for threshold in loss_thresholds:
                if loss_hits[threshold] is not None:
                    results[threshold].append(loss_hits[threshold])
                else:
                    results[threshold].append(iterations)  # Mark as max if not hit

    return results

# Plotting function
def plot_results(results, loss_thresholds):
    """Plot best, worst, and average iterations for each specific loss threshold."""
    averages = []
    best_cases = []
    worst_cases = []

    # Process results for each threshold
    for threshold in loss_thresholds:
        iterations = results[threshold]  # Iterations at this specific threshold
        averages.append(np.mean(iterations))  # Average iteration count
        #best_cases.append(np.min(iterations))  # Best-case iteration (minimum)
        #worst_cases.append(np.max(iterations))  # Worst-case iteration (maximum)

    print(averages)

    # Plot the results: X-axis = Iterations, Y-axis = Loss Thresholds
    plt.figure(figsize=(10, 6))

    #plt.plot(best_cases, loss_thresholds, marker='o', linestyle='--', label="Best Case (Min)")
    plt.plot(averages, loss_thresholds, marker='o', linestyle='-', label="Average Iterations")
    #plt.plot(worst_cases, loss_thresholds, marker='o', linestyle='-.', label="Worst Case (Max)")

    plt.ylabel("Loss Threshold")
    plt.xlabel("Iterations to Reach Threshold")
    plt.title("Iterations to Reach Loss Thresholds (Average)")
    plt.legend()
    plt.grid(True)
    plt.show()

def plot2():

# Data
    import matplotlib.pyplot as plt

    # Data
    iterations = [1000, 750, 500, 250, 100]  # Iterations (increasing)
    new = [3314.608333333333,4437.533333333334,5472.975,7508.341666666666,8745.841666666667]

    #even = [2588.1066666666666, 5955.86, 11870.225, 27139.02, 41783.778333333335]
    #loss = [1288.6066666666666, 2044.13, 3743.1116666666667, 9870.288333333334, 28285.40166666667]
    #gain = [10197.421666666667, 15645.133333333333, 24579.305, 37200.28, 45696.443333333336]

    # Plot
    plt.figure(figsize=(10, 6))
    plt.plot(new, iterations, marker='o', linestyle='-', label="Moderation")
    #plt.plot(loss, iterations, marker='o', linestyle='--', label="Loss")
    #plt.plot(gain, iterations, marker='o', linestyle='-.', label="Gain")

    # Labels and Title
    plt.xlabel("Iterations")
    plt.ylabel("Threshold Values")
    plt.title("Dinners")
    plt.legend()
    plt.grid(True)

    # Show the plot
    plt.show()


import time

# Main execution
def main():
    """Main function to run loss threshold tracking and plotting."""
    # Generate varied macros
    print(ideal_macros)
    random_macros = generate_random_macros(ideal_macros, n=40)
    
    # Define loss thresholds and iterations
    loss_thresholds = [1000, 750, 500, 250, 100]
    iterations = 1
    
    # Use "Dinner" meal combinations for testing
    meal_combos = meal_combinations["Dinner"]

    # Measure effectiveness
    start_time = time.time()
    results = measure_effectiveness(meal_combos, random_macros, iterations, loss_thresholds)
    end_time = time.time()
    # Plot results

    # Calculate total time and average time per iteration
    total_time = end_time - start_time
    print(len(meal_combinations["Dinner"]))
    avg_time_per_iteration = total_time / (iterations * len(meal_combinations["Dinner"]))

    for threshold in loss_thresholds:
        helper = results[threshold]  # Iterations at this specific threshold
        print(np.mean(helper))

    # Print results
    print(f"\nTotal Time for {iterations} iterations: {total_time:.2f} seconds")
    print(f"Average Time Per Iteration: {avg_time_per_iteration:.6f} seconds")
    
    plot_results(results, loss_thresholds)

def loss():
    macros1 = {
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
    macros = {
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
    res =calculate_loss(macros, macros1)
    print(res)

    # [1000, 750, 500, 250, 100]
    #10 calories = 100 loss
    #10 carbs = 100 loss



if __name__ == "__main__":
    #loss()
    main()
