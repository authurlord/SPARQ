import pandas as pd

df = pd.read_csv('table.csv')
# Convert viewers (in millions) to numeric, handling any potential parsing issues
viewers = pd.to_numeric(df['viewers (in millions)'], errors='coerce')
# Drop any invalid entries (if any)
viewers = viewers.dropna()
# Calculate the average viewership
average_viewership = viewers.mean()
print(f"Final Answer: {average_viewership:.2f}")