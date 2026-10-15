import pickle

# Load the features
with open("features.pkl", "rb") as f:
    data = pickle.load(f)

# Swap the IDs: 20231399512 ↔ 9999
old_ids = data["ids"]
new_ids = []
for id_ in old_ids:
    if id_ == "20231399512":
        new_ids.append("9999")
    elif id_ == "9999":
        new_ids.append("20231399512")
    else:
        new_ids.append(id_)

data["ids"] = new_ids

# Save
with open("features.pkl", "wb") as f:
    pickle.dump(data, f)

print("✅ Swapped! Mitchell ↔ Somto")
print(f"   Old: {old_ids}")
print(f"   New: {new_ids}")