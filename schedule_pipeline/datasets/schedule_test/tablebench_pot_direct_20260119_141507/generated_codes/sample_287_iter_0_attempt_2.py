import pandas as pd
import re

df = pd.read_csv('table.csv')

# Clean the data: extract numeric values from strings like "19.0 (1996)"
def extract_numeric(value):
    match = re.search(r'(\d+\.?\d*)', value)
    if match:
        return float(match.group(1))
    return 0.0

# Select relevant columns: 1996 general and 2004 european
cols_1996 = '1996 general'
col_2004 = '2004 european'

# Extract rows (regions) and their values
data = []
for idx, row in df.iterrows():
    # Get 1996 general value
    val_1996 = extract_numeric(row[cols_1996])
    # Get 2004 european value
    val_2004 = extract_numeric(row[col_2004])
    
    # Skip if either value is 0 or invalid
    if val_1996 == 0 or val_2004 == 0:
        continue
    
    # Compute percentage change
    change = ((val_2004 - val_1996) / val_1996) * 100
    data.append(change)

# Compute average percentage change
avg_change = sum(data) / len(data) if data else 0

print(f"Final Answer: {avg_change:.2f}")