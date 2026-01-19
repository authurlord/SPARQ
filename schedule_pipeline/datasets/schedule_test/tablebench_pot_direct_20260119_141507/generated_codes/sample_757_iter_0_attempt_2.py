import pandas as pd

df = pd.read_csv('table.csv')
# Filter rows for seasons 1 to 6 (index 0 to 5)
seasons_1_to_6 = df[df['season'].between(1, 6)]
# Extract viewership values
viewership_values = seasons_1_to_6['us viewers (millions)'].astype(float)
# Calculate mean as forecast
forecast_viewership = viewership_values.mean()
print(f"Final Answer: {forecast_viewership:.2f}")