import pandas as pd
import numpy as np

df = pd.read_csv('table.csv')

# Clean 'capacity in use' column: remove any non-numeric characters and convert to float
df['capacity in use'] = df['capacity in use'].str.replace('%', '').str.replace(',', '.').astype(float)

# Calculate the correlation coefficient between 'total passengers' and 'capacity in use'
correlation = df['total passengers'].corr(df['capacity in use'])

print(f"Final Answer: {correlation:.2f}")