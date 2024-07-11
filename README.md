# Rebar Optimization

This project optimizes the cutting of rebars to minimize waste using linear programming and the column generation method. The main goal is to reduce the amount of waste generated when cutting standard-length rebars into specified lengths for construction projects.

## Table of Contents

- [Rebar Optimization](#rebar-optimization)
- [Table of Contents](#table-of-contents)
- [Installation](#installation)
- [Usage](#usage)
  - [Running the Script](#running-the-script)
- [Example](#example)
- [Acknowledgements](#acknowledgements)
- [License](#license)


## Installation

1. Clone the repository:

    ```sh
    git clone https://github.com/yourusername/Rebar_Optimization.git
    cd Rebar_Optimization

2. Set up a virtual environment and activate it:
    ```sh
    python -m venv env
    source env/bin/activate  # On Windows use `env\Scripts\activate`

3. pip install -r requirements.txt

## Usage

### Running the Script

To run the script, use the following command:

    python rebar_optimization.py <file_path> -t <file_type>
    

1. <file_path>: Path to the input Excel/CSV file.
2. <file_type>: Type of the input file (either excel or csv).

For example, to run the script with an Excel file:

    python rebar_optimization.py Data/sample_data.xlsx -t excel

For a CSV file:

    python rebar_optimization.py Data/sample_data.xlsx -t csv

## Example

The sample_data.xlsx file contains the following columns:

1. Label: The label for the rebar.
2. Count: The quantity of the rebar.
3. Bar Length: The length of the rebar.

The script will output the optimized cutting instructions and total waste.

## Acknowledgements

1. https://towardsdatascience.com/column-generation-in-linear-programming-and-the-cutting-stock-problem-3c697caf4e2b - Provided code ideas and guidance.




