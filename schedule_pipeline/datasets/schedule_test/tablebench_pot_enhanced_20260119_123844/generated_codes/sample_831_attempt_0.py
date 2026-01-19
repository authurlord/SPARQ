import pandas as pd
import numpy as np

df = pd.read_csv('table.csv')
# Filter out the 'total' row
df_filtered = df[df['year'] != 'total']
# Convert 'tries' column to numeric
df_filtered['tries'] = pd.to_numeric(df_filtered['tries'])
# Calculate variance
variance_tries = np.var(df_filtered['tries'], ddof=1)  # Using sample variance (ddof=1)
print(f"Final Answer: {variance_tries:.2f}")