import pandas as pd
import numpy as np

# Load the data
df = pd.read_csv('table.csv')

# Identify the rows that represent ward data (first block)
# The row with ['ward', 'lyrette', 'maguire', ...] is a header for the second block
# We need to extract only the first block of data (before the second header)

# The first row of data starts with 'orlãans'
# We'll filter out the row that contains the second header (contains 'ward' as first element and has the list of names)
# The actual ward data rows start with ward names like 'orlãans', 'innes', etc.

# Extract only the rows where the first value is a ward name (not the header row)
# Remove the row that has 'ward' as the first element and contains the list of names
ward_data_rows = df[df.iloc[:, 0].str.contains(r'^[a-zA-Z\s]+$', na=False) & (~df.iloc[:, 0].str.contains('ward', case=False, na=False))]

# Select the 'bello' column (index 1)
bello_values = ward_data_rows.iloc[:, 1].astype(str).str.replace(r'\D', '', regex=True).str.strip()

# Convert to numeric, ignoring errors
bello_numeric = pd.to_numeric(bello_values, errors='coerce')

# Drop NaNs
bello_numeric = bello_numeric.dropna()

# Calculate mean and standard deviation
mean_bello = bello_numeric.mean()
std_bello = bello_numeric.std()

print(f"Final Answer: {mean_bello:.1f}, {std_bello:.1f}")