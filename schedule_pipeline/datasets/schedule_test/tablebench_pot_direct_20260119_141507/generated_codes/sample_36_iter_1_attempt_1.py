import pandas as pd

df = pd.read_csv('table.csv')
# Ensure the 'tumbling' column is numeric and compute the mean
tumbling_scores = pd.to_numeric(df['tumbling'], errors='coerce')
average_tumbling = tumbling_scores.mean()
print(f"Final Answer: {average_tumbling:.1f}")