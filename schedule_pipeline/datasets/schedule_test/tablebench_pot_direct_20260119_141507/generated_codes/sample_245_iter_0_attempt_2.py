import pandas as pd

# Load the data
df = pd.read_csv('table.csv')

# Extract the 'Poland and Danzig' column (index 1) and convert to float
poland_danzig = df.iloc[:, 1].astype(float)

# Calculate the increase from 1934 to 1939
increase = poland_danzig.iloc[5] - poland_danzig.iloc[0]

print(f"Final Answer: {increase:.1f}")