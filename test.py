import pickle

with open("../data_hdqi/Cliff_TYPE1_m500n50k2_t1.pkl", "rb") as f:
    loaded_list = pickle.load(f)

print(loaded_list)

