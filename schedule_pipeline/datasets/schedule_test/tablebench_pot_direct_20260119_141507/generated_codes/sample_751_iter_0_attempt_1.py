import pandas as pd

# Load the data
df = pd.read_csv('table.csv')

# Extract the 'tower division' column and corresponding years
years = df['year'].astype(int)
tower_division = df['tower division']

# Convert to a list of values from 1801 to 1871
values = tower_division[years.between(1801, 1871)].tolist()
years_subset = years[years.between(1801, 1871)].tolist()

# Calculate the linear trend (slope) from 1801 to 1871
n = len(years_subset)
if n < 2:
    raise ValueError("Not enough data points to compute trend.")

# Linear regression: y = mx + b
# m = (sum((x_i - x_mean)(y_i - y_mean))) / (sum((x_i - x_mean)^2))
x = [year - 1801 for year in years_subset]  # offset to start at 0
y = values

x_mean = sum(x) / n
y_mean = sum(y) / n

numerator = sum((xi - x_mean) * (yi - y_mean) for xi, yi in zip(x, y))
denominator = sum((xi - x_mean) ** 2 for xi in x)

slope = numerator / denominator
intercept = y_mean - slope * x_mean

# Project to year 1881 (i.e., x = 80 years after 1801)
x_1881 = 80
projected_value = slope * x_1881 + intercept

print(f"Final Answer: {int(projected_value)}")