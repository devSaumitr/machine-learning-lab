class std :
     def __init__(self) :
          self.name = "saumitra"


s1 = std()
print(id(s1))
s2 = s1 
print(id(s2))
print(s1.name) 
s2.name = "adarsh"
print(s1.name)
print(s2.name)



class Person :
     def __init__(self , name , gen) :
         self.name = name 
         self.gender = gen 
         
                  
# p.greet() ;         
def greet(self) :
         print(id(self))
         print(f"my name is {self.name} and i am {self.gender} . ")
         x = Person("ankit" , "MALE")
         self.name = "Ram"
         return  x ;     
p = Person("Saumitra" , "Male") ;
print(greet(p)) 
X = greet(p)    

print(id(p))
print(greet(X))



p1 = Person("raj" , 323)

p2 = Person("raj" , 3423)

p3 = Person("raj" , 34324)

p4 = Person("raj" , 3423442)


p = [  p1 , p2 ,p3 ,p4 ]
for i in p :
     print(i.gender)