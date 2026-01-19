import pandas as pd
import numpy as np

df = pd.read_csv('table.csv')
# Filter rows where Division is 'Third Division North'
third_division_north = df[df['Division'] == 'Third Division North']
# Extract goals scored in the 'Goals' column (from 'Apps' and 'Goals' columns)
goals = third_division_north['Goals'].astype(int)
# Calculate variance
variance = np.var(goals, ddof=1)  # Using sample variance
print(f"Final Answer: {variance:.2f}")