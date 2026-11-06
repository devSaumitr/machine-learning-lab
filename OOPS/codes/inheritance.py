class User :
     def __init__( self , name , age  ):
         self.age = age 
         self.name = name 
      
      
     def login(self):
           return (" successfully logged in ")           
          
          
class student(User) : 
    # def __init__(self ,  rollno , name , age ) :
    #      self.rollno = rollno
    #      self.name = name 
    #      self.age = age
         
    
    def enroll (self  ):
        self.login()
        

    def display_enrollment_det(self) :
        print(f"{self.login()} \n {self.name} \n {self.age} ")
                
                
s = student ( "saumitra" , 23 )
# s.enroll("saumitra" , 23  )

s.display_enrollment_det()