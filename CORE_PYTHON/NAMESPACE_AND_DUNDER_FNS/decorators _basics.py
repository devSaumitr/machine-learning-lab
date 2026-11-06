def quad_power( func , num ) :
     return func(num)**2


def sq(num ):
    return num **2 


print(quad_power(sq , 2))


a = quad_power 
print(a(sq , 2 ))


#  in python we can pass function as the argument in the  other function

def my_dec(func):

    def wrapper():
        print("***********")
        func()
        print("***********")

    return wrapper


def hello():
    print("hello")

a = my_dec(hello)

a()


#  child can  acceses parents property  even after the parent has dies 
 
def my_Dec (func) :
        
    def wrapper ( ) :  
        print("hi")
        func (); 
        print("bye")
    
    return wrapper     


def func() :
    print("my name is  saumitra " )


a = my_Dec(func)    
a()


@my_Dec
def hello():
    print("HOW's THE JOSH")
    
    
hello()    