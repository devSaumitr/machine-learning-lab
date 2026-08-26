from Person import Person as p
import pickle

person = p("Saumitra", "Awasthi", 23, "M")

with open("person.pkl", "wb") as f:
    pickle.dump(person, f)

print("Object pickled successfully")