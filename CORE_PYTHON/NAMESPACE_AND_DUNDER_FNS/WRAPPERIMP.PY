import time
def dec(fn): 
    def wrapper (* nums ):
        start = time.time()
        print(f" total marks is : {sum(nums)} and their average is : {fn(*nums)}" )
        end = time.time() ;
        print(f"total time to calcualte total marks is : {end - start}") 
        
    return wrapper 


@dec 
def fn(*nums ):
    return sum(nums) / len(nums)


fn( 1 ,2,3,4,5,6)

