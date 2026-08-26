class dog :
     def sound(self) :
          print("dog barks");
class lion :
      def sound (self) : 
           print("lion roars") ; 


def make_Sound( animal ): 
     animal.sound()


l = lion()
d = dog(); 
make_Sound(d);
make_Sound(l);



class animal : 
     def sound (self):
          print("animal sound ") ;
class dog( animal) : 
     def sound(self):
          print("dog barks ");
class cat (animal) :
      def sound(self) :
            print("cat meows") ;
class lion (animal) :
     pass


an = dog();
an.sound() ;
an = cat() ;
an.sound();

an = lion();
an.sound();


class complex :
    def __init__(self , real , img  ):
          self.real = real ; 
          self.img = img ;
    def display(self) : 
         print( str (self.real) + "i" + " + " + str (self.img) + "j")
def add_2complex( c1 , c2 ):
     rp = c1.real + c2.real ; 
     ip = c1.img + c2.img ;
     c = complex(rp ,ip );
     return c ;


c1 = complex(10 ,40)
c1.display();
c2 = complex(13,10)
c2.display();
print(add_2complex(c1, c2 ).display())




class complex :
      def __init__(self , real ,img ):
           self.real = real
           self.img = img ;
      def __add__(self, other  ):
            rp = self.real + other.real 
            ip = self.real + other.img 
            
           












