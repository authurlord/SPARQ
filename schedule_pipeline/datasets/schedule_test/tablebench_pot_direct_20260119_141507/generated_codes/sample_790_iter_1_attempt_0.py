import pandas as pd
import numpy as np

df = pd.read_csv('table.csv')
# Filter out rows where 'average' is 'n / a' and convert to float
average_scores = df['average'].astype(str).str.replace('n / a', '').dropna().astype(float)
# Calculate standard deviation
std_dev = np.std(average_scores)
print(f"Final Answer: {std_dev:.2f}")