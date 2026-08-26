#  namespace is the area where python stores the varaible and object dictionary so wha u can do
# LEBG is the rule pyhton uses to search the varable 
# local -> enclosing -> global -> built in 

# namespace  means space where the variable are defined 

# scope decides where the objects can be used  
#   you can not edit the  global objects inside the local scope 

#  when global and local varaibles having same name :
    #  they can exist in th differerent scope
    
# GLOBAL ATTRUBUTES CAN BE EDITED INSIDE THE LOCAL BUT ITS NOT THE GOOD PRACTICE TO DO 
    # its done by denoting the declaring the varaible as the global  in the local space  
        
#  YOU CAN CREATE THE GLOBAL FROM LOCAL
    # its done by denoting the declaring the varaible as the global in the local space  

# PARAMETERS ARE ALWAYS LOCAL

              
# a = 2 

    # a = 3 

# def fn():
#     global a 
#     a = 3       
#     print (a)
    
# fn()



# built in scope 

# functions that are the part of the pythons 
#  import builtinis  


# import builtins
# print(dir(builtins))
# give all the builtins ;


def max( n ) :
    print("hello")
    
max(12);     


#  check always done fron LEGB rule

        #       x
        #       ↓
        # ┌───────────┐
        # │ Local     │
        # └─────┬─────┘
        #       ↓ not found
        # ┌───────────┐
        # │ Enclosing │
        # └─────┬─────┘
        #       ↓ not found
        # ┌───────────┐
        # │ Global    │
        # └─────┬─────┘
        #       ↓ not found
        # ┌───────────┐
        # │ Built-in  │
        # └─────┬─────┘
        #       ↓ not found
        #   NameError
        
        # nonlocal  tells the compiler you want to make changes in the outer or the enclosing
        
        
        # heirarchy of the LEGB rule 
        
        
        # it goes in to out firstly search in the 
        # local -> enclosing 1 -> enclosing 2->enclosing 3 -> global -> built in 
        
def ennc ():
    n = 23
    def enc1 ():
        nonlocal n 
        n += 2
        
    enc1()        
    print(n) ;     
ennc()        