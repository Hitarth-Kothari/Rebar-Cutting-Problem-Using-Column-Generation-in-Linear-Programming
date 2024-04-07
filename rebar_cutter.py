# Using code from https://towardsdatascience.com/column-generation-in-linear-programming-and-the-cutting-stock-problem-3c697caf4e2b
import numpy as np
import pandas as pd
from scipy.optimize import linprog
from fractions import Fraction
import argparse

STANDARD_REBAR_LENGTH = 240

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
    df['Joining Instructions'] = None
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
                df.at[index, 'Joining Instructions'] = f"Join {additional_rebars} standard bars and one {remaining_length}\" piece"
            else:
                df.at[index, 'Joining Instructions'] = f"Use one {remaining_length}\" piece"
    return df, total_additional_rebars

def solve_knapsack(total_width, widths, duals):
    try:
        assert widths.shape[0] == duals.shape[0], "widths and duals dimension mismatch"
        solution = linprog(-duals, A_ub=np.atleast_2d(widths), b_ub=np.atleast_1d(total_width), bounds=(0, np.inf), method='highs', options={"disp": False}, integrality=1)
        return solution
    except Exception as e:
        print(f"Error in solve_knapsack: {e}")
        raise

def process_and_display_results(solution, patterns_matrix, labels, lengths, rebar_id_start, quantities_needed, waste_pieces, cutting_instructions):
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
                    length_count = min(length_count, quantities_needed[label] // pattern_use_count) if quantities_needed[label] // pattern_use_count > 0 else length_count
                    cut_instruction = f"  Cut {length_count} pieces of {lengths[length_index]} inches for {label}"
                    cuts.append(cut_instruction)
                    waste -= length_count * lengths[length_index]
                    quantities_needed[label] -= length_count * pattern_use_count
            instruction += f" | Waste: {waste*pattern_use_count} inches"
            cutting_instructions.append((instruction, cuts))
            total_waste += waste * pattern_use_count
            waste_pieces.append([rebar_id, waste, pattern_use_count])
            rebar_id += pattern_use_count

    return rebar_id, total_waste, quantities_needed, waste_pieces, cutting_instructions

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

def try_using_waste(labels, lengths, quantities_needed, waste_pieces, cutting_instructions, total_waste):
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
    
    return total_waste, quantities_needed, waste_pieces, cutting_instructions

def wrapper_optimization_loop(df, chunk_size=56):
    df, total_additional_rebars = preprocess_dataframe(df)
    total_waste_accumulated = 0
    rebar_id_start = 1
    waste_pieces = []
    cutting_instructions = []
    final_verification = {label: 0 for label in df['Label'].unique()}  # Tracking produced quantities

    # Create chunks of the dataframe based on chunk_size
    chunks = [df[i:i+chunk_size] for i in range(0, df.shape[0], chunk_size)]

    for chunk in chunks:
        print ('Optimizing a Chunk')
        labels = chunk['Label'].values
        lengths = chunk['Bar Length'].values
        quantities = chunk['Count'].values.astype(int)
        quantities_needed = dict(zip(labels, quantities))

        if len(set(labels)) != len(labels):
            print('The labels in the Dataset are not unique within a chunk. Please Fix')
            continue

        # Initialize or update quantities_needed for global tracking
        for label, quantity in zip(labels, quantities):
            final_verification[label] -= quantity  # Deduct the required quantity initially

        temp = total_waste_accumulated
        print('Using wastes')
        # Try using waste from previous chunks (You need to define this function)
        total_waste_accumulated, quantities_needed, waste_pieces, cutting_instructions= try_using_waste(labels, lengths, quantities_needed, waste_pieces, cutting_instructions, total_waste_accumulated)
        temp = temp - total_waste_accumulated
        print(f"Saved {temp} inches of waste")

        # Process each chunk individually
        while not all(value <= 0 for value in quantities_needed.values()):
            solution, patterns_matrix, updated_labels, updated_lengths = main_optimization(labels, lengths, list(quantities_needed.values()))
            new_rebar_id_start, total_waste, quantities_needed, waste_pieces, cutting_instructions = process_and_display_results(solution, patterns_matrix, updated_labels, updated_lengths, rebar_id_start, quantities_needed, waste_pieces, cutting_instructions)
            total_waste_accumulated += total_waste
            rebar_id_start = new_rebar_id_start  # Update for the next chunk

        # Update final verification for produced quantities
        for label, quantity in zip(labels, quantities):
            produced_qty = quantity - quantities_needed.get(label, 0)
            final_verification[label] += produced_qty  # Add back the produced quantity
        print ('Finished Optimizing a Chunk')

    print_cutting_instructions(cutting_instructions)
    print_final_verification(df, final_verification)

    total_rebars_used = rebar_id_start - 1 + total_additional_rebars
    print(f"\nOverall Total Waste: {total_waste_accumulated} inches")
    print(f"Total Standard Rebars Needed: {total_rebars_used}")

def print_cutting_instructions(cutting_instructions):
    for instruction, cuts in cutting_instructions:
        print(instruction)
        for cut in cuts:
            print(cut)

def print_final_verification(df, final_verification):
    print("\nFinal Verification and Joining Instructions:")
    for label, delta_qty in final_verification.items():
        original_qty = df[df['Label'] == label]['Count'].sum()
        produced_qty = original_qty + delta_qty  # Adjust based on delta from processing
        print(f"{label}: Required {original_qty}, Produced {produced_qty}, {'Met' if produced_qty >= original_qty else 'Not Met'}")
        joining_instructions = df[df['Label'] == label]['Joining Instructions'].iloc[0]
        if pd.notnull(joining_instructions):
            print(f"  - {joining_instructions}")

def main():
    # Setup command-line interface
    # parser = argparse.ArgumentParser(description="Process Excel/CSV file for Rebar Optimization.")
    # parser.add_argument("file_path", type=str, help="Path to the input Excel/CSV file.")
    # parser.add_argument("-t", "--file_type", type=str, choices=['excel', 'csv'], default="excel", help="Type of the input file (excel or csv).")
    
    # Parse arguments
    # args = parser.parse_args()
    file_path = 'rebar_file.csv'#args.file_path
    file_type = 'csv'#args.file_type

    try:
        # Load data and preprocess
        if file_type == 'excel':
            df = pd.read_excel(file_path, engine='openpyxl', usecols=['Label', 'Count', 'Bar Length'])
        elif file_type == 'csv':
            df = pd.read_csv(file_path, usecols=['Label', 'Count', 'Bar Length'])
        else:
            raise ValueError("Unsupported file type. Use 'excel' or 'csv'.")

        df.dropna(subset=['Count'], inplace=True)
        df = df[df['Count'] > 0]
        df['Bar Length'] = df['Bar Length'].apply(fraction_to_decimal)
        
        # Run the optimization wrapper function
        wrapper_optimization_loop(df)
    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    main()