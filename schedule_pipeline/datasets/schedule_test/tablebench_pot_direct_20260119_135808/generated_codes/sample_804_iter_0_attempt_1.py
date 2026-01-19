import pandas as pd

df = pd.read_csv('table.csv')
# Convert 'rating' column to numeric
df['rating'] = pd.to_numeric(df['rating'], errors='coerce')
# Calculate mean and median
mean_rating = df['rating'].mean()
median_rating = df['rating'].median()
print(f"Final Answer: {mean_rating:.2f}, {median_rating:.2f}")