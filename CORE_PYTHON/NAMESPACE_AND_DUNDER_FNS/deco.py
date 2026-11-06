import time
def timer(func):
    def wrapper():
        start = time.time()
        print("hello")
        func()
        print("time taken by " , func.__name__ , time.time()- start )
    return wrapper


def timer (func):
    def wrapper (*args) :
        start = time.time()
        return f" {func(args)} and time taken to run {func.__name__ }is {time.time()-start} secs"
    return wrapper ;



@timer
def total(*nums):
    return (sum(nums))

print(total(3,4,5))

import time

def return_time(func):

    def wrapper(*args):
        start = time.time()

        result = func(*args)

        return f"{result} and time taken to run {func.__name__}: {time.time() - start} secs"

    return wrapper


@return_time
def total(*nums):
    return sum(nums)

print(total(3, 4, 5))