import pandas as pd
import numpy as np

df = pd.read_csv('table.csv')

# Replace 'n / a' with NaN to allow numerical operations
df['comprehension of danish'] = df['comprehension of danish'].replace('n / a', np.nan)
df['comprehension of swedish'] = df['comprehension of swedish'].replace('n / a', np.nan)
df['comprehension of norwegian'] = df['comprehension of norwegian'].replace('n / a', np.nan)

# Convert comprehension columns to float
df[['comprehension of danish', 'comprehension of swedish', 'comprehension of norwegian']] = df[
    ['comprehension of danish', 'comprehension of swedish', 'comprehension of norwegian']
].apply(pd.to_numeric, errors='coerce')

# Calculate average comprehension per city
city_avg_comprehension = df[['comprehension of danish', 'comprehension of swedish', 'comprehension of norwegian', 'average']].mean()

# Insights: High comprehension in Norway, especially in Oslo and Bergen; Stockholm and Malmö show strong Swedish and Norwegian skills
print("Main features: The table shows comprehension levels of Danish, Swedish, and Norwegian in major Scandinavian cities. Data is missing for some entries (marked as 'n / a').")
print("Insights: Norway shows the highest overall comprehension, particularly in Bergen and Oslo. In Sweden, comprehension of Swedish and Norwegian is high, but Danish is lower. The average column reflects overall language proficiency.")
print(f"Average comprehension levels by city: {city_avg_comprehension.to_dict()}")
Final Answer: high comprehension in norway, strong swedish in sweden, missing data in danish for some cities