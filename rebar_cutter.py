# Using code from https://towardsdatascience.com/column-generation-in-linear-programming-and-the-cutting-stock-problem-3c697caf4e2b
import numpy as np
import pandas as pd
from scipy.optimize import linprog
from fractions import Fraction

STANDARD_REBAR_LENGTH = 240
FILE_PATH = 'rebar_file.xlsx'

def fraction_to_decimal(value):
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
    df['Welding Instructions'] = None
    total_additional_rebars = 0
    for index, row in df.iterrows():
        if row['Bar Length'] > STANDARD_REBAR_LENGTH:
            additional_rebars = row['Bar Length'] // STANDARD_REBAR_LENGTH
            total_additional_rebars += additional_rebars * row['Count']
            remaining_length = row['Bar Length'] % STANDARD_REBAR_LENGTH
            if remaining_length == 0:
                remaining_length = STANDARD_REBAR_LENGTH
                total_additional_rebars -= row['Count']
                additional_rebars -= 1
            df.at[index, 'Bar Length'] = remaining_length
            if additional_rebars > 0:
                df.at[index, 'Welding Instructions'] = f"Weld {additional_rebars} standard bars and one {remaining_length}\" piece"
            else:
                df.at[index, 'Welding Instructions'] = f"Use one {remaining_length}\" piece"
    return df, total_additional_rebars

def solve_knapsack(total_width, widths, duals):
    return linprog(-duals, A_ub=np.atleast_2d(widths), b_ub=np.atleast_1d(total_width), bounds=(0, np.inf), method='highs', options={"disp": False}, integrality=1)

def process_and_display_results(solution, patterns_matrix, labels, lengths, rebar_id_start, quantities_needed):
    cutting_instructions = []
    total_waste = 0
    waste_pieces = []
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
                    quantities_needed[label] -= length_count * pattern_use_count
            instruction += f" | Waste: {waste*pattern_use_count} inches"
            cutting_instructions.append((instruction, cuts))
            total_waste += waste * pattern_use_count
            waste_pieces.append([rebar_id, waste, pattern_use_count])
            rebar_id += pattern_use_count

    # Process to utilize waste pieces after all cuts are made
    for i in range(len(waste_pieces)):
        rebar_id = waste_pieces[i][0]
        waste_piece = waste_pieces[i][1]
        count = waste_pieces[i][2]
        updated_cuts = cutting_instructions[i][1] if cutting_instructions[i][1] is not None else []
        update_waste_instruction = cutting_instructions[i][0] if cutting_instructions[i][0] is not None else []
        for label, length in zip(labels, lengths):
            if length <= waste_piece and quantities_needed[label] > 0:
                num_cuts_from_waste = int(waste_piece // length)
                if num_cuts_from_waste > 0:
                    instruction = f"  Cut {num_cuts_from_waste} pieces of {length} inches for {label}"
                    updated_cuts.append(instruction)
                    waste_piece -= length * num_cuts_from_waste
                    total_waste -= length * num_cuts_from_waste*count
                    quantities_needed[label] -= num_cuts_from_waste*count
                    waste_pieces[i][0] = waste_piece  # Update the remaining waste piece after cutting
                    update_waste_instruction = f"\nRebar {rebar_id} to {rebar_id + count - 1}:" + f" | Waste: {waste_piece*count} inches"

        # Update the cutting instructions with the additional cuts from waste
        if updated_cuts:
            cutting_instructions[i] = (update_waste_instruction, updated_cuts)

    for instruction, cuts in cutting_instructions:
        print(instruction)
        for cut in cuts:
            print(cut)

    return rebar_id, total_waste, quantities_needed

def main_optimization(labels, lengths, quantities):
    quantities = np.array(quantities)  # Convert quantities to a numpy array
    patterns_matrix = np.eye(len(lengths)) * (STANDARD_REBAR_LENGTH // lengths)
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
    quantities_needed = dict(zip(labels, quantities))
    rebar_id_start = 1
    total_waste_accumulated = 0

    while not all(value <= 0 for value in quantities_needed.values()):
        solution, patterns_matrix, updated_labels, updated_lengths = main_optimization(labels, lengths, list(quantities_needed.values()))
        rebar_id_start, total_waste, quantities_needed = process_and_display_results(solution, patterns_matrix, updated_labels, updated_lengths, rebar_id_start, quantities_needed)
        total_waste_accumulated += total_waste

    print("\nFinal Verification and Welding Instructions:")
    for index, row in df.iterrows():
        label = row['Label']
        original_qty = row['Count']
        produced_qty = original_qty - quantities_needed.get(label, 0)
        print(f"{label}: Required {original_qty}, Produced {produced_qty}, {'Met' if produced_qty >= original_qty else 'Not Met'}")
        if pd.notnull(row['Welding Instructions']):
            print(f"  - {row['Welding Instructions']}")

    total_rebars_used = rebar_id_start + total_additional_rebars
    print(f"\nOverall Total Waste: {total_waste_accumulated} inches")
    print(f"Total Standard Rebars Needed: {total_rebars_used}")

# Assuming df is preprocessed and ready.
file_path = FILE_PATH
df = pd.read_excel(file_path, engine='openpyxl', usecols=['Label', 'Count', 'Bar Length'])
df.dropna(subset=['Count'], inplace=True)
df = df[df['Count'] > 0]
df['Bar Length'] = df['Bar Length'].apply(fraction_to_decimal)

wrapper_optimization_loop(df)
