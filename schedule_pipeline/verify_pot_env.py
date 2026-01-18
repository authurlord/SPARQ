import sys
import os
import pandas as pd
import json

# Add parent dir to path to import utils
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.python_executor import execute_python_code

def run_test(name, table_dict, question, code):
    print(f"--- Test: {name} ---")
    print(f"Question: {question}")
    
    # Create DF
    df = pd.DataFrame(table_dict['data'], columns=table_dict['columns'])
    print(f"Data Shape: {df.shape}")
    
    # Run Code
    print("Executing Code...")
    result = execute_python_code(code, df)
    print(f"Result:\n{result}")
    
    if "Execution Error" in result:
        print("[FAILED] Execution Error detected.")
    else:
        print("[SUCCESS] executed without crashing.")
    print("\n")

def main():
    print("Verifying TableBench PoT Environment with 5 Samples...\n")

    # Sample 1: Handling Non-Numeric Strings in Numeric Columns
    # Question: "What is the average value of the 2001 general election across all regions in Italy?"
    s1_table = {
        "columns": ["Region", "Party", "2001 general"],
        "data": [
            ["piedmont", "with fi", "3.5"],
            ["lombardy", "with fi", "3.4"],
            ["veneto", "with fi", "5.0"],
            ["sicily", "with fi", "14.4"]
        ]
    }
    s1_code = """
import pandas as pd
df = pd.read_csv('table.csv')
# '2001 general' contains numbers as strings
# Ensure we convert to numeric
mean_val = pd.to_numeric(df['2001 general'], errors='coerce').mean()
print(f"Final Answer: {mean_val:.2f}")
"""
    run_test("Sample 1 (Numeric Parsing)", s1_table, "Average of 2001 general", s1_code)

    # Sample 2: Handling Special Strings (e.g., 'Current') and Date Math
    # Question: "What is the total number of years served by all mayors?"
    s2_table = {
        "columns": ["Mayor", "Taking Office", "Leaving"],
        "data": [
            ["Vivian Burrill", "1901", "1902"],
            ["Michel Angers", "2009", "Current"]
        ]
    }
    s2_code = """
import pandas as pd
import datetime

df = pd.read_csv('table.csv')

# Handle 'Current' - assuming current year 2014 or similar for dataset context, 
# or perhaps we should strip it. 
# For safety, let's coerce 'Current' to NaN and drop or fill with 2014.
current_year = 2014 

# Convert columns to numeric
df['start'] = pd.to_numeric(df['Taking Office'], errors='coerce')
df['end'] = df['Leaving'].apply(lambda x: current_year if str(x).lower() == 'current' else x)
df['end'] = pd.to_numeric(df['end'], errors='coerce')

# Calculate duration
df['duration'] = df['end'] - df['start']
total_years = df['duration'].sum()

print(f"Final Answer: {total_years}")
"""
    run_test("Sample 2 (Date/String Logic)", s2_table, "Total years served", s2_code)

    # Sample 3: Filtering and summation
    # Question: "Total length of rivers flowing into left side"
    s3_table = {
        "columns": ["Name", "Position", "Length"],
        "data": [
            ["Feudinge", "left", "6.3"],
            ["Ilse", "right", "8.4"],
            ["Wetschaft", "left", "29.0"]
        ]
    }
    s3_code = """
import pandas as pd
df = pd.read_csv('table.csv')

# Filter for Position == 'left'
left_rivers = df[df['Position'].str.lower() == 'left']

# Sum length
# Length might be string
total_len = pd.to_numeric(left_rivers['Length'], errors='coerce').sum()
print(f"Final Answer: {total_len}")
"""
    run_test("Sample 3 (Filter & Sum)", s3_table, "Total length left side", s3_code)

    # Sample 4: Percentage Parsing
    # Question: "Average percentage of national votes"
    s4_table = {
        "columns": ["year", "% of national vote"],
        "data": [
            ["1956", "39.7%"],
            ["1959", "41.2%"],
            ["1986", "38.58%"]
        ]
    }
    s4_code = """
import pandas as pd
df = pd.read_csv('table.csv')

# Strip '%' and convert to float
# Note: '39.7%' -> 39.7
vals = df['% of national vote'].astype(str).str.rstrip('%').astype(float)
avg = vals.mean()
print(f"Final Answer: {avg:.2f}%")
"""
    run_test("Sample 4 (Percentage Parsing)", s4_table, "Average percentage", s4_code)

    # Sample 5: Simple Aggregation
    # Question: "Average number of podiums achieved by a team in a single season"
    s5_table = {
        "columns": ["Team", "Podiums"],
        "data": [
            ["Team A", "1"],
            ["Team B", "1"],
            ["Team C", "6"],
            ["Team D", "0"]
        ]
    }
    s5_code = """
import pandas as pd
df = pd.read_csv('table.csv')
# Each row is a team in a season
avg_podiums = pd.to_numeric(df['Podiums']).mean()
print(f"Final Answer: {avg_podiums}")
"""
    run_test("Sample 5 (Simple Stats)", s5_table, "Average podiums", s5_code)

if __name__ == "__main__":
    main()
