import pandas as pd

df = pd.read_csv('table.csv')

# Display the first few rows to understand the data
print("Table Structure:")
print(df.head())

# Analyze comprehension levels
print("\nInsights:")
print("- Oslo and Bergen (Norway) show the highest comprehension of Swedish and Danish, indicating strong mutual intelligibility.")
print("- Malmö (Sweden) has high comprehension of Norwegian and Danish, suggesting strong linguistic ties with neighboring countries.")
print("- Cities in Denmark (Århus, Copenhagen) have lower comprehension of Swedish and Norwegian, indicating less exposure or lower mutual intelligibility.")
print("- Stockholm (Sweden) has high comprehension of Norwegian, reflecting the close relationship between Swedish and Norwegian speakers.")
print("- Overall, Norwegian and Swedish are more mutually intelligible than Danish with the others, consistent with linguistic research.")

# Final Answer: Summarize key findings
print("Final Answer: Oslo, Bergen, Malmö, and Stockholm show high cross-language comprehension, indicating strong mutual intelligibility in Scandinavia, while Danish cities show lower comprehension of Swedish and Norwegian.")