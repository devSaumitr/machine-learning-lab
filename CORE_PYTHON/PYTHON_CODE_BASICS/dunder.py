# class student:
#      def __init__(self , name , roll ) :
#          self.name = name ;
#          self.roll = roll ;
#      def __str__(self):
#           return "Student" + "(Name = " + self.name + " , " + "Roll = " + str (self.roll) + ")"
          
# s = student("Adarsh" ,101)
# print(s)
         
        


# class Team :
#      def __init__(self , members) : 
#           self . members = members ;
#      def __len__(self ) :
#           return len(self.members);

# t = Team(["Virat", "Rohit", "Gill"])
# print(len(t))



# class Complex :
#     def __init__(self , real , imaginary ):
#           self.real = real 
#           self.imaginary = imaginary;
#     def __add__( self , other) :
#            rp = self.real + other.real
#            ip = self.imaginary + other.imaginary
#            return Complex(rp, ip);



#     def __str__(self) : 
#          return  str(self.real ) +" + " + str(self.imaginary) + "j"; 

    

# c1 = Complex(2, 3)
# c2 = Complex(4, 5)

# c3 = c1 + c2
# print(c3)
    
         
# class Student :
#     def __init__ (self , name , roll ) :
#            self.name = name 
#            self.roll = roll;
                        
#     def __eq__(self , other ): 
#          if (self.roll == other.roll) :
#               return True ;
#          else :
#                 return False               
         
# s1 = Student("Adarsh", 101)
# s2 = Student("Rahul", 101)

# print(s1 == s2)         




class Book :
    def __init__(self , name , pages ):
         self.Name = name 
         self.Pages = pages
        

    def __str__ (self):
         return "Book : " + self.Name + "," + "Pages : " +str( self.Pages ) ;
    def __len__(self):
         return (self.Pages);
    def __eq__(self, other):
            return self.Name == other.Name;



b1 = Book("Python", 300)
b2 = Book("DSA", 250)

print(b1)
print(len(b1))
print(b1 == b2)



























