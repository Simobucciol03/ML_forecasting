import pickle

# Apri il file in modalità lettura binaria
with open("/Users/simonebucciol/Desktop/project/machine_learning/file_brain/rf_model.pkl", "rb") as f:
    data = pickle.load(f)

# Stampa il contenuto
print(data)
