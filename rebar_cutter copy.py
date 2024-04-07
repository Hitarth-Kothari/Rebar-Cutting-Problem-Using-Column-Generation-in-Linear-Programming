#using code from https://towardsdatascience.com/column-generation-in-linear-programming-and-the-cutting-stock-problem-3c697caf4e2b
import numpy as np
import pandas as pd
from scipy.optimize import linprog
from fractions import Fraction

STANDARD_REBAR_LENGTH = 240  # Convert standard rebar length from feet to inches

def fraction_to_decimal(value):
    if isinstance(value, str):
        value = value.replace('"', '').strip()  # Remove trailing inches symbol and whitespace
        if ' ' in value:
            whole_number, fraction_part = value.split(' ')
            return float(whole_number) + float(Fraction(fraction_part))
        else:
            return float(Fraction(value))
    else:
        return float(value)
    
def preprocess_dataframe(df):
    """Preprocess the DataFrame to fit cutting requirements."""
    df['Bar Length'] = df['Bar Length'].apply(fraction_to_decimal)
    for index, row in df.iterrows():
        if row['Bar Length'] > STANDARD_REBAR_LENGTH:
            remaining_length = row['Bar Length'] % STANDARD_REBAR_LENGTH
            if remaining_length == 0:
                remaining_length = STANDARD_REBAR_LENGTH
            df.at[index, 'Bar Length'] = remaining_length
    return df

file_path = r'C:\Users\hitu1\Desktop\Coding\mahdis\rebar_file.xlsx'
df = pd.read_excel(file_path, engine='openpyxl', usecols=['Label', 'Count', 'Bar Length'])
df = preprocess_dataframe(df)

df.dropna(subset=['Count'], inplace=True)  # Remove rows with NaN in 'Count'
df = df[df['Count'] > 0]  # Keep rows with positive 'Count' values only

df['Bar Length'] = df['Bar Length'].apply(fraction_to_decimal)

required_lengths = df['Bar Length'].values  # Now in inches
required_quantities = df['Count'].values.astype(int)
labels = df['Label'].values

max_cuts_per_length = np.floor(STANDARD_REBAR_LENGTH / required_lengths).astype(int)
patterns_matrix = np.eye(len(required_lengths)) * max_cuts_per_length
cost_vector = np.ones(len(required_lengths))  # Objective function to minimize the number of rebars used

def solve_knapsack(total_width, widths, duals):
    return linprog(
        -duals, A_ub=np.atleast_2d(widths), b_ub=np.atleast_1d(total_width),
        bounds=(0, np.inf), method='highs', options={"disp": False}, integrality=1
    )

initial_solution = linprog(cost_vector, A_ub=-patterns_matrix, b_ub=-required_quantities, bounds=(0, None), method='highs', options={"disp": False})

if initial_solution.success:
    if hasattr(initial_solution, 'ineqlin') and initial_solution.ineqlin is not None:
        duals = -initial_solution.ineqlin.marginals
    else:
        print("Exiting due to lack of inequality constraints.")
        exit()
else:
    print(f"Initial LP solution was not successful: {initial_solution.message}")
    exit()

for _ in range(1000):
    knapsack_solution = solve_knapsack(STANDARD_REBAR_LENGTH, required_lengths, duals)
    if 1 + knapsack_solution.fun < -1e-4:
        new_pattern = knapsack_solution.x
        patterns_matrix = np.hstack((patterns_matrix, new_pattern.reshape((-1, 1))))
        cost_vector = np.append(cost_vector, 1)
        initial_solution = linprog(cost_vector, A_ub=-patterns_matrix, b_ub=-required_quantities, bounds=(0, None), method='highs')
    else:
        break

final_solution = linprog(cost_vector, A_ub=-patterns_matrix, b_ub=-required_quantities, bounds=(0, np.inf), method='highs', integrality=1)

total_produced = np.dot(patterns_matrix, final_solution.x)
total_rebars_used = np.sum(final_solution.x)

print(f"Total Rebars Used: {int(total_rebars_used)}")
for label, length, quantity, produced in zip(labels, required_lengths, required_quantities, total_produced):
    print(f"Label: {label}, Length: {length} inches, Demand: {quantity}, Produced: {int(produced)}")

if np.all(total_produced >= required_quantities):
    print("All demands are met.")
else:
    print("Not all demands are met. Adjust the optimization model or parameters.")

rebar_id = 1
for pattern_index, pattern_use_count in enumerate(final_solution.x.astype(int)):
    if pattern_use_count > 0:
        print(f"\nRebar {rebar_id} to {rebar_id + pattern_use_count - 1}:")
        for length_index, length_count in enumerate(patterns_matrix[:, pattern_index].astype(int)):
            if length_count > 0:
                print(f"  Cut {length_count} pieces of {required_lengths[length_index]} inches for {labels[length_index]}")
        rebar_id += pattern_use_count