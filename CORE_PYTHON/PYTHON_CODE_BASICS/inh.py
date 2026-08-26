# class Person :
#     name = "anonymous"
    
#     # def changeName( self , name) : 
#     #     self.name = name ;
#     @classmethod
#     def change( cls , name) : 
#         cls.name = name

# p1 = Person() 
# p1.change("Saumitra"); 
# print(p1.name)
# print(Person.name)
 

# class student : 
#     school = " abc"
#     def __init__(self , p , c ,m):
#         self.p = p ; self.c = c; self.m = m ; 
#         # self.percentage = str ((self.p +self. c +self.  m ) / 3) + "  %" ;

#     # def calc_per(self ) : 
#     #     self.per = str ((self.p +self. c +self.  m ) / 3) + "  %" ;

#     @property
#     def percentage(self) :
#         return   str ((self.p +self. c +self.  m ) / 3) + "  %"
# s = student(89,88,83 )
 
# print(s.percentage)
# s.p = 94 
# print(s.percentage)
# #  u are actually  using the member functions as the member variable
 

# class student : 
#     def __init__(self , marks ) :
#          self.marks =marks ; 
#     @property
#     def percentage (self) : 
#         return self.marks/10 ;


# s = student(987);
# print(s.percentage) 
        
         
# property acts as getter and u can also use attribute.setter for setting up the values 


class student :   
    def __init__( self) : 
        self.marks = 0 ;


    @marks.setter 
    def marks ( self  ,val)  : 
        if( val < 0 ) : 
             print ( " marks cannot be negative") 
        else : 
                          
            self.marks = val ; 


    @property
    def percentage  ( self ) :
         return self.marks/10;




s = student();
s.marks =  927
print(s.percentage);

