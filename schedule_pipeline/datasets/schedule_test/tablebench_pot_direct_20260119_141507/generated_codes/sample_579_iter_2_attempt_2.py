import pandas as pd
import re

df = pd.read_csv('table.csv')

# Clean the 'year' column: extract integers from strings like "2010 , 2012"
def extract_years(year_str):
    years = []
    if isinstance(year_str, str):
        # Split by comma and space and extract integers
        parts = re.split(r',\s*', year_str)
        for part in parts:
            if part.strip().isdigit():
                years.append(int(part.strip()))
    return years

# Apply the function to get all years in each row
df['years'] = df['year'].apply(extract_years)

# Filter rows where any of the extracted years is in 2000–2007
filtered_rows = df[df['years'].apply(lambda x: any(y >= 2000 and y <= 2007 for y in x))]

# Calculate the average quantity of buses in the filtered rows
average_quantity = filtered_rows['quantity'].mean()

print(f"Final Answer: {average_quantity:.1f}")