import pandas as pd

df = pd.read_csv('table.csv')

# Extract the relevant values for Constitutional Democratic Party (Kadets)
first_duma_seats = 179
second_duma_seats = 92
total_first_duma = 566
total_second_duma = 453

# Calculate percentages
first_percent = (first_duma_seats / total_first_duma) * 100
second_percent = (second_duma_seats / total_second_duma) * 100

# Calculate the change
seat_share_change = second_percent - first_percent

print(f"Final Answer: {seat_share_change:.2f}")