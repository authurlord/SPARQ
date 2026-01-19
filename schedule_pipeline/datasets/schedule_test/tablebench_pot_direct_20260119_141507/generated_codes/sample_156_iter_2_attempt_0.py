import pandas as pd
import numpy as np

# Load the table
df = pd.read_csv('table.csv')

# Convert relevant columns to numeric, handling any parsing issues
df['population in millions'] = pd.to_numeric(df['population in millions'], errors='coerce')
df['area km 2'] = pd.to_numeric(df['area km 2'], errors='coerce')
df['area % of eu'] = pd.to_numeric(df['area % of eu'].str.rstrip('%').replace({'%': '', ' ': ''}), errors='coerce')
df['pop density people / km 2'] = pd.to_numeric(df['pop density people / km 2'], errors='coerce')
df['population % of eu'] = pd.to_numeric(df['population % of eu'].str.rstrip('%').replace({'%': '', ' ': ''}), errors='coerce')

# Drop rows with NaN after conversion
df = df.dropna()

# Compute correlation with 'population % of eu'
correlations = df[['area km 2', 'area % of eu', 'pop density people / km 2', 'population % of eu']].corr()['population % of eu'].abs()

# Find factors with correlation > 0.3 (considered significant)
significant_factors = correlations[correlations > 0.3].index.tolist()

if len(significant_factors) == 0:
    print("Final Answer: no clear impact")
else:
    print(f"Final Answer: {', '.join(significant_factors)}")