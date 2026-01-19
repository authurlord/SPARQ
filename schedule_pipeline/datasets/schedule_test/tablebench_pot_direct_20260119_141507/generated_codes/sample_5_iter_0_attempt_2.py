import pandas as pd
import re

df = pd.read_csv('table.csv')

# Extract the '2001 general' column
column_2001 = df['2001 general']

# Clean values: remove text in parentheses and convert to float
cleaned_values = []
for val in column_2001:
    # Remove everything in parentheses
    cleaned = re.sub(r'\([^)]*\)', '', val)
    # Convert to float, handling cases like 'with fi'
    if cleaned.strip() == 'with fi':
        continue
    try:
        cleaned_values.append(float(cleaned.strip()))
    except ValueError:
        continue

# Calculate average
average_2001 = sum(cleaned_values) / len(cleaned_values) if cleaned_values else 0
print(f"Final Answer: {average_2001:.2f}")