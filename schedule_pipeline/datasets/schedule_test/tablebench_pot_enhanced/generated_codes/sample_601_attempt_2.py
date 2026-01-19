import pandas as pd

df = pd.read_csv('table.csv')

# Extract data for 1931
start_r_class = df[df['Year'] == '1931']['R class in service at start of year'].values[0]
withdrawn = df[df['Year'] == '1931']['Quantity withdrawn'].values[0]

# Calculate end of year R class
end_r_class = int(start_r_class) - int(withdrawn)
print(f"Final Answer: {end_r_class}")