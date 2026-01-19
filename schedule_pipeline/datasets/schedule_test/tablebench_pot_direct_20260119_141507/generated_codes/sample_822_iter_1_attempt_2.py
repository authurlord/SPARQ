import pandas as pd
import numpy as np

df = pd.read_csv('table.csv')
# Convert 'us viewers (millions)' to numeric, handling any parsing issues
viewers = pd.to_numeric(df['us viewers (millions)'], errors='coerce')
# Calculate average and standard deviation
mean_viewers = viewers.mean()
std_viewers = viewers.std()
print(f"Final Answer: {mean_viewers:.2f}, {std_viewers:.2f}")