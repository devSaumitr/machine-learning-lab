# m1 = input ( "movie_1 : "  )
# m2 = input ("movie_2 : "  )
# m3 = input ("movie_3 : "  )
# list =[];
# list.append(m1)
# list.append(m2)
# list.append(m3)
# print(list);

lst = [1, 2 ,3 ,4,5,4,3,2,1];
# i, j, flag  = 0 , len(lst) -1 , 1 ; 
# while( i<=j) :
#     if(lst[i] == lst[j]):
#         i+=1
#         j -=1
#     else :
#          print( " list is not palindrome") ; 
#          flag = 0 ; 
#          break ; 
# if (flag) :
#     print("list is palindrome") ; 


lt = lst.copy() ; 
lt.reverse() ; 
if( lst == lt ):
    print(lst, " is palindrome")
else :
    print(lst, " is  not palindrome")
         
