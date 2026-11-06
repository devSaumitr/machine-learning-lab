# square = lambda x : x *x ; 
# res = list(map( square , lt));


# lt = [3 ,4 ,5,6,7]
# add = lambda *nums : sum(nums) 
# res = list(map(add , lt ,lt ,lt ))
# # print(res); 
# nums = list(input("enter nums").split())
# nums = list(map(int  , nums))
# print (type(nums[0])) 


values = [0, 1, "", "hello", False, True, None, 5]

result = list(filter( int , values))

print(result)