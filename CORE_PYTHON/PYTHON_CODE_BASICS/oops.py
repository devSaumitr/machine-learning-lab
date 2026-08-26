class student : 
    college = "GL bajaj";
    name = "anonymous"
    def __init__(self , roll , rank , marks ,  name = None) :
          print(  " data being stored in the data base ") 
          
          if name is not None :
              self.name =  name
          self.marks = marks
          self.roll = roll 
          self.rank = rank;
    def marks_Avg ( self  ) : 
         total = 0 
         for  m in self.marks :
              total += m;
         avg = total/len (self.marks) 
         return avg ;

    def display( self ) : 
         print(self.name ,"\n",self.roll ,"\n" ,self.rank)
         print(self.college)
         print(" average of all subjects is : " , self.marks_Avg())



def inp_marks()  :
     marks= (input( " enter marks : ")).split()
     
     for i  in range(len(marks)) :
          marks[i] =  int ( marks[i])
     return marks       
     
s1 = student(int(input(" enter roll no. :")) , 
             int ( input( "enter rank : ")) , 
             inp_marks() , 
             input(" enter your name : ")  )
s1.display();
#  default constructors are being called by the python if not created by programmer 
#  def __init__( self) :
        # pass




        #  objects occupy space in the memory 

        