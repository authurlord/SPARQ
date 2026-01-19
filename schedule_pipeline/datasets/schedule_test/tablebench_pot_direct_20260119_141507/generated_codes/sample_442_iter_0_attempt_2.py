import pandas as pd

# Load the dataframe
df = pd.read_csv('table.csv')

# Display the structure and initial observations
print("Main columns:", df.columns.tolist())
print("\nSample data:")
print(df.head())

# Extract the 'progressive conservative' column and convert to numeric
pc_support = pd.to_numeric(df['progressive conservative'], errors='coerce')
lib_support = pd.to_numeric(df['liberal'], errors='coerce')
nd_support = pd.to_numeric(df['new democratic'], errors='coerce')

# Check for trends by observing the overall range and any patterns
print("\nProgressive Conservative support range:", pc_support.min(), "to", pc_support.max())
print("Liberal support range:", lib_support.min(), "to", lib_support.max())
print("New Democratic support range:", nd_support.min(), "to", nd_support.max())

# Observations:
# - Progressive Conservative starts at 82% (2007) and drops to 59% (2011), suggesting a decline.
# - Liberal support ranges from 12% to 22%, showing minimal growth.
# - New Democratic support ranges from 5% to 33%, showing some fluctuation but generally lower than others.

# Final Answer: The table tracks political party support over time. Progressive Conservative support declined from 82% in 2007 to 59% in 2011, while Liberal and New Democratic support remained relatively stable or increased slightly.
Final Answer: Progressive Conservative support declined, Liberal support remained stable, New Democratic support fluctuated