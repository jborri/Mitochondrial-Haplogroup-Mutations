
import pandas as pd

def load_haplogroups(filename):
    df = pd.read_csv(filename)
    haplogroups = {}
    for index, row in df.iterrows():
        haplogroups[row['Haplogroup']] = row['Defining_Mutations'].split()
    return haplogroups

haplogroups = load_haplogroups('Haplogroup Analysis Tool/Haplogroups.csv')

def identify_haplogroup(mutations, haplogroups):
    for haplogroup, defining_mutations in haplogroups.items():
        if all(mutation in mutations for mutation in defining_mutations):
            return haplogroup
    return "Unknown Haplogroup"

user_mutations = input("Enter the mutations separated by space: ").split()
haplogroup = identify_haplogroup(user_mutations, haplogroups)
print(f"The haplogroup is: {haplogroup}")