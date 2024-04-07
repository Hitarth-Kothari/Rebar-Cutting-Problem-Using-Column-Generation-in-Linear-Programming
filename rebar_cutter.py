import numpy as np
import pandas as pd
from scipy.optimize import linprog
from fractions import Fraction

# Global configuration for standard rebar length in inches
STANDARD_REBAR_LENGTH = 240

def fraction_to_decimal(value):
    """Converts string fractions to decimal."""
    if isinstance(value, str):
        value = value.replace('"', '').strip()
        if ' ' in value:
            whole_number, fraction_part = value.split(' ')
            return float(whole_number) + float(Fraction(fraction_part))
        else:
            return float(Fraction(value))
    else:
        return float(value)

def preprocess_dataframe(df):
    df['Additional Standard Rebars'] = 0
    df['Welding Instructions'] = None  # Initialize a column for welding instructions
    total_additional_rebars = 0
    for index, row in df.iterrows():
        if row['Bar Length'] > STANDARD_REBAR_LENGTH:
            # Calculate how many standard rebars are needed for this length
            additional_rebars = int(row['Bar Length'] // STANDARD_REBAR_LENGTH)
            total_additional_rebars += additional_rebars * row['Count']  # Account for multiple counts
            remaining_length = row['Bar Length'] % STANDARD_REBAR_LENGTH
            if remaining_length == 0:
                remaining_length = STANDARD_REBAR_LENGTH
                total_additional_rebars -= row['Count']  # Adjust for exact divisions
                additional_rebars -= 1
            df.at[index, 'Bar Length'] = remaining_length
            
            # Store welding instructions
            if additional_rebars > 0:
                df.at[index, 'Welding Instructions'] = f"Weld {additional_rebars} standard bars and one {remaining_length}\" piece"
            else:
                df.at[index, 'Welding Instructions'] = f"Use one {remaining_length}\" piece"
                
    return df, total_additional_rebars


def solve_knapsack(total_width, widths, duals):
    """Solves the knapsack problem for given parameters."""
    return linprog(-duals, A_ub=np.atleast_2d(widths), b_ub=np.atleast_1d(total_width),
                    bounds=(0, np.inf), method='highs', options={"disp": False}, integrality=1)

def process_and_display_results(solution, patterns_matrix, labels, lengths, rebar_id_start):
    cutting_instructions = []
    label_cut_quantities = {label: 0 for label in labels}  # Track cut quantities by label
    total_waste = 0
    rebar_id = rebar_id_start
    for pattern_index, pattern_use_count in enumerate(solution.x.astype(int)):
        if pattern_use_count > 0:
            instruction = f"\nRebar {rebar_id} to {rebar_id + pattern_use_count - 1}:"
            waste = STANDARD_REBAR_LENGTH
            cuts = []
            for length_index, length_count in enumerate(patterns_matrix[:, pattern_index].astype(int)):
                if length_count > 0:
                    label = labels[length_index]
                    cut_instruction = f"  Cut {length_count} pieces of {lengths[length_index]} inches for {label}"
                    cuts.append(cut_instruction)
                    waste -= length_count * lengths[length_index]
                    label_cut_quantities[label] += length_count * pattern_use_count
            instruction += f" | Waste: {waste} inches"
            cutting_instructions.append((instruction, cuts))
            total_waste += waste * pattern_use_count
            rebar_id += pattern_use_count

    # Print cutting instructions for this loop
    for instruction, cuts in cutting_instructions:
        print(instruction)
        for cut in cuts:
            print(cut)
    return rebar_id, total_waste, label_cut_quantities

def main_optimization(labels, lengths, quantities):
    patterns_matrix = np.eye(len(lengths))*(STANDARD_REBAR_LENGTH//lengths)  # Initial patterns
    cost_vector = np.ones_like(lengths)
    sol = linprog(cost_vector, A_ub=-patterns_matrix, b_ub=-quantities, bounds=(0, None), method='highs')
    for _ in range(1000):
        duals = -sol.ineqlin.marginals
        price_sol = solve_knapsack(STANDARD_REBAR_LENGTH, lengths, duals)
        if 1 + price_sol.fun < -1e-4:
            patterns_matrix = np.hstack((patterns_matrix, price_sol.x.reshape((-1, 1))))
            cost_vector = np.append(cost_vector, 1)
            sol = linprog(cost_vector, A_ub=-patterns_matrix, b_ub=-quantities, bounds=(0, None), method='highs')
        else:
            break
    solution = linprog(cost_vector, A_ub=-patterns_matrix, b_ub=-quantities, bounds=(0, np.inf), method='highs', integrality=1)
    return solution, patterns_matrix, labels, lengths

def wrapper_optimization_loop(df):
    df, total_additional_rebars = preprocess_dataframe(df)
    labels = df['Label'].values
    lengths = df['Bar Length'].values
    quantities = df['Count'].values.astype(int)
    rebar_id_start = 1
    total_waste_accumulated = 0
    cut_record = {label: 0 for label in labels}  # Initialize cutting record

    while not all(quantities <= 0):
        solution, patterns_matrix, updated_labels, updated_lengths = main_optimization(labels, lengths, quantities)
        rebar_id_start, total_waste, label_cut_quantities = process_and_display_results(solution, patterns_matrix, updated_labels, updated_lengths, rebar_id_start)
        total_waste_accumulated += total_waste

        # Update quantities and cut_record based on what was actually cut
        for i, label in enumerate(labels):
            cut_qty = label_cut_quantities.get(label, 0)
            quantities[i] -= cut_qty
            cut_record[label] += cut_qty

        if not any(quantities > 0):  # Break if all demands are met
            break

    # Final Verification and Display Welding Instructions
    print("\nFinal Verification and Welding Instructions:")
    for index, row in df.iterrows():
        label = row['Label']
        original_qty = row['Count']
        produced_qty = cut_record.get(label, 0)
        print(f"{label}: Required {original_qty}, Produced {produced_qty}, {'Met' if produced_qty >= original_qty else 'Not Met'}")
        
        # Print welding instructions if available
        if pd.notnull(row['Welding Instructions']):
            print(f"  - {row['Welding Instructions']}")
    
    # Print summary
    total_rebars_used = rebar_id_start - 1 + total_additional_rebars
    print(f"\nOverall Total Waste: {total_waste_accumulated} inches")
    print(f"Total Standard Rebars Needed: {total_rebars_used}")

# Assuming df is preprocessed and ready.
# Example of loading data and preprocessing
file_path = 'rebar_file.xlsx'
df = pd.read_excel(file_path, engine='openpyxl', usecols=['Label', 'Count', 'Bar Length'])
df.dropna(subset=['Count'], inplace=True)
df = df[df['Count'] > 0]
df['Bar Length'] = df['Bar Length'].apply(fraction_to_decimal)

# Execute the optimization wrapper
wrapper_optimization_loop(df)