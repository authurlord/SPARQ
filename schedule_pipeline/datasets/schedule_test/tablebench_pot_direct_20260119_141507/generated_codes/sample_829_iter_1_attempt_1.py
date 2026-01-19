import pandas as pd
import numpy as np

# Load the data
df = pd.read_csv('table.csv')

# Identify valid ward rows: those where the first element is a ward name (not a councillor name)
# Councillor names start with letters like 'lyrette', 'maguire', etc.
valid_ward_rows = df[df['ward'].str.contains(r'^[a-zA-Z]+$', na=False) & (~df['ward'].str.contains(r'lyrette|maguire|o\'brien|pita|ryan|st arnaud|scharf|taylor|watson|wright', case=False))]

# Extract the 'bello' column from valid rows
bello_values = valid_ward_rows['bello'].astype(str).str.strip().str.replace(',', '').astype(float)

# Calculate mean and standard deviation
mean_bello = np.mean(bello_values)
std_bello = np.std(bello_values)

print(f"Final Answer: {mean_bello:.2f}, {std_bello:.2f}")