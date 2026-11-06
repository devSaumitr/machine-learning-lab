# def outer(func):
#     def inner (*nums ) :
#         print(sum(nums))
#     return inner

# def func(*nums):
#     print(sum(nums))
    
# a = outer(func);
# print(a(12 ,12 ,12 ))    

# @outer
# def fn(*nums):
#     print(sum(nums))

# fn(2,2,2,2,2,2,2,2)    

def decorator(func):

    def wrapper():
        print("Before")
        func()
        print("After")

    return wrapper()

def fn():
    print('geiorge')
    
a = decorator(fn)
    