from Person import Person as p
import json


# def show_object(person):
#     if isinstance(person, p):
#         return {
#             "fname": person.fname,
#             "lname": person.lname,
#             "age": person.age,
#             "gender": person.gender
#         }

    
    
def show_object(person):
    if isinstance(person, p):
        return f" name -> {person.fname}  { person.lname}  age -> {person.age} gender  -> {person.gender}"
            

person = p("Saumitra", "Awasthi", 23, "M")


with open("obj.json", "w") as f:
    json.dump(person, f, default=show_object, indent=4)
    
    
    
    
    
    
    
    
    
    
     # now the time for concept of the pickeling 
    #  
     
      