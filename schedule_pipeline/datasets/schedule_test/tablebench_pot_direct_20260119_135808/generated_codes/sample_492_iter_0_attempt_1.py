import pandas as pd
df = pd.read_csv('table.csv')
df = df[df['Round'] != 'Round']
round_order = {'1st': 1, '2nd': 2, '3rd': 3}
df['Round_num'] = df['Round'].map(round_order)
round_totals = {}
for round_name in df['Round'].unique():
    round_data = df[df['Round'] == round_name]
    mz_deputies = 0
    mz_senators = 0
        if row['Miloš Zeman'] != '-' and row['Miloš Zeman'] != '':
            mz_deputies += int(row['Miloš Zeman'])
        if row['Miloš Zeman_1'] != '-' and row['Miloš Zeman_1'] != '':
            mz_senators += int(row['Miloš Zeman_1'])
    total_deputies = 0
    total_senators = 0
        if row['Václav Klaus'] != '-' and row['Václav Klaus'] != '':
            total_deputies += int(row['Václav Klaus'])
        if row['Václav Klaus_1'] != '-' and row['Václav Klaus_1'] != '':
            total_senators += int(row['Václav Klaus_1'])
        if row['Jaroslava Moserová'] != '-' and row['Jaroslava Moserová'] != '':
            total_deputies += int(row['Jaroslava Moserová'])
        if row['Jaroslava Moserová_1'] != '-' and row['Jaroslava Moserová_1'] != '':
            total_senators += int(row['Jaroslava Moserová_1'])
        if row['Miloš Zeman'] != '-' and row['Miloš Zeman'] != '':
            total_deputies += int(row['Miloš Zeman'])
        if row['Miloš Zeman_1'] != '-' and row['Miloš Zeman_1'] != '':
            total_senators += int(row['Miloš Zeman_1'])
    total_possible = total_deputies + total_senators
    mz_total = mz_deputies + mz_senators
        print(f"Majority win in {round_name}")
        print(f"Zeman: {mz_total}, Total: {total_possible}, %: {mz_total / total_possible * 100:.1f}%")
    print("No majority win found.")