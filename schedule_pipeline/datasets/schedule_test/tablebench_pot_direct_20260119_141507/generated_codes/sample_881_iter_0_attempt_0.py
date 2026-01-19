import pandas as pd

# Load the data
df = pd.read_csv('table.csv', index_col=False)

# Convert relevant columns to float (the values are strings with decimal points)
df = df.astype({col: float for col in df.columns if col not in ['-', 'Soviet Union', 'Poland and Danzig', 'Finland', 'Estonia', 'Latvia', 'Lithuania']})

# Extract the years 1934 and 1939
year_1934 = df.iloc[0:6, 1:].values  # Rows 0 to 5, columns from 1 onwards
year_1939 = df.iloc[5, 1:].values    # Row 5 (1939), columns from 1 onwards

# Find the country with the highest import in 1939
max_1939_index = year_1939.argmax()
country_names = ['Soviet Union', 'Poland and Danzig', 'Finland', 'Estonia', 'Latvia', 'Lithuania']
country_name = country_names[max_1939_index]

# Get the values for 1939 and 1934
value_1939 = year_1939[max_1939_index]
value_1934 = year_1934[max_1939_index]

# Calculate the difference
difference = value_1939 - value_1934

print(f"Final Answer: {country_name}, {difference:.1f}")