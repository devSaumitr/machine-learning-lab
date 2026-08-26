from Person import Person as p
import pickle

with open("person.pkl", "rb") as f:
    d = pickle.load(f)

d.display_info()