import pandas as pd
import numpy as np

# Load the data
df = pd.read_csv('table.csv')

# Clean the data: extract numeric values from columns corresponding to 1996 and 2004
# Columns of interest: '1996 general' and '2004 european'
# We will parse each row to extract the value for 1996 and 2004

# Create a list to store percentage changes
percentage_changes = []

# Iterate over each row (region)
for idx, row in df.iterrows():
    # Extract 1996 value (from '1996 general' column)
    try:
        val_1996 = float(row['1996 general'].replace('(', '').replace(')', '').strip())
    except:
        val_1996 = np.nan
    
    # Extract 2004 value (from '2004 european' column)
    try:
        val_2004 = float(row['2004 european'].replace('(', '').replace(')', '').strip())
    except:
        val_2004 = np.nan
    
    # Skip if either value is missing or invalid
    if pd.isna(val_1996) or pd.isna(val_2004):
        continue
        
    # Calculate percentage change
    change = ((val_2004 - val_1996) / val_1996) * 100
    percentage_changes.append(change)

# Compute average percentage change
avg_change = np.mean(percentage_changes) if percentage_changes else 0

print(f"Final Answer: {avg_change:.2f}")