# // what my task is ? 
#  i need to make an decorater that checks whether the given fn recives the argunments of the same type 


# check for the input like check for the imteger 
def my_dec (data_type ):
    
    def outer_wrapper( func):
        def inner_wrapper(* args):
            if( data_type == type(args[0])):
                return func (*args)  
            else :
                return f"this type of coversion nit possible "
        return inner_wrapper
    
    return outer_wrapper    


@my_dec(int)
def tot_marks(*args):
    return sum(args)

print(tot_marks (12 ,23 ))

