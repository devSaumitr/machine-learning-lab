
# students = {
#     1: "Adarsh",
#     2: "Rahul",
#     3: "Aman",
#     4: "Priya",
#     5: "Neha",
#     6: "Rohan",
#     7: "Ankit",
#     8: "Pooja",
#     9: "Karan",
#     10: "Simran"
# }

# print(students[1])
# students[1]= "Saumitra"
# print(students[1])

data = {
      "person1" : { "name" : "saumitra" ,
                    "age" : 22 , 
                    "marks" : { 
                                "phy" : 98 , "chem" : 89 , "maths" : 95
                                  } 
      }
        ,
       "person2" : { "name" : "anushka ", "age" : 22 },
       "person3" : { "name" : "raj" , "age" : 22 , "marks" : { 
          "phy" : 98 , "chem" : 89 , "maths" : 95 } }
        }

# print(data["person3"]);
# t = tuple(data.keys())
# tt =list(data.keys())
# lt = data.keys() ; 
# print (tt)
# print(t)
# print (lt)
# vl = data.values() ; 
# vlt = tuple(data.values() ); 
# vlll = list(data.values()) ; 
# print(vl ,"\n"); 
# print(vlt ,"\n");
# print(vlll,"\n");
print(data.get("person1"))