import pandas as pd

df = pd.read_csv('table.csv')
# Filter rows for White/Blue and Red/Blue
filtered_df = df[df['Color'].isin(['White/Blue', 'Red/Blue'])]
# Find the color with higher Pin (Tip) value
max_pin = filtered_df.loc[filtered_df['Pin (Tip)'].idxmax()]
print(f"Final Answer: {max_pin['Color']}")