def sum(*nums) :
    x = 0 ; 
    print(type(nums))
    for ele in nums :
        x  = x + 10
    return x       
fg
x = sum(1, 2,3,4,5,6)
print(x)


def data (**kwargs) :
    for items in kwargs.items() :
        print( items[0] , '=' , items[1] ) 

data (name  = 'saumitra' , 
      age = 23 ,
      ambition = '+infintiy')


def sq (x):
    return x**2 
square = lambda x : x** 2 ;
print(square(5));
# lambda arguments: expression