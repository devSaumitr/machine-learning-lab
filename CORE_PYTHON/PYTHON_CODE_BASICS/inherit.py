# class phone :
#     def connect_call(self):
#         print("call is being connected ")
#     def disconnect_call(self) :
#         print("call is being disconnected")
# class smartphone (phone):
#     def click_photo (self):
#         print("photo is being clicked") ;
#     def multimedia (self) :
#         print("watch movies ");
# class Iphone (smartphone):
#      def security(self)  : 
#          print("security is being maintained ") ;
#      def service (self) : 
#          print("service is taken care off ") ;            
# x = Iphone();
# x.service()
# x.connect_call();
# x.click_photo(); 










# class camera :
#     def capture(self) : 
#         print("photo is being captured");
# class speaker : 
#     def sound(self) :
#          print("sound is being produced ") ; 
# class phone ( camera , speaker) :
#     def connect_call(self):
#         print("call is being connected ")
#     def disconnect_call(self) :
#         print("call is being disconnected")
    

# p = phone(); 
# p.sound();
# p.capture();p.connect_call();






#   now learnig the useage of the super 


class Person:
    def __init__(self, name  , age ):
        self.name = name
        self.__age = age 

    def Name(self):
        print("Name is :", self.name)


class Student(Person):
    def __init__(self, name, roll, dep , age ):
        super().__init__( name  , age )
        self.roll = roll
        self.dep = dep

    def rollno(self):
        print("Roll no. is :", self.roll)

    def department(self):
        print("Department is :", self.dep)
    def display( self) : 
         self.Name() 
         self.department()
         self.rollno();
         print(self._Person__age)


s = Student("Saumitra", 21, "AIML" ,32)

s.display()
# s.rollno()
# s.department()






 