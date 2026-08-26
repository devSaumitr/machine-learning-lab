#  sigmoid 
# formula = 1 / (1 + e^( -x))
import numpy as np ;
# def sigmoid (arr):
#     return 1   / (1 + np.exp(-arr)) ;


# a = np.arange( 1 , 10  )
# print(sigmoid(a)) ; 




# #  mean square error 


# #  markqs based on iq 



# # def MSE ( arr) : 
# #     ans = 0 ; 
# #     md = np.mean(arr)
# #     for i in range(arr.size) : 
# #         tp = (arr[i]- md)**2
# #         ans +=  tp ; 

# #     ans /= arr.size;
# #     return ans ;




# pred =np.random.randint(1 , 50 , 25); 



# act = np.random.randint(1 , 50 , 25 ); 



# def MSE( act , pred  ) : 
#     ans =0 
#     for  i in range( act.size) :
#         tp = (act - pred)** 2
#         ans += tp ;  
#         tp = 0 ;
#     return ans / act.size;


# def MSE ( act , pred ) : 

#     return (( act  - pred )** 2).sum()/ act.size;
    
# def MSE ( act , pred import numpy as np
import matplotlib.pyplot as plt

x = np.linspace(-10, 10, 10, dtype=int)
y = x

plt.plot(x, y)
plt.show()) : 



#     return np.mean((( act  - pred )** 2)) ;

# def BCE ( act , pred ) :
#    return -( np.mean ( (act * np.log(pred)) + ( ( 1 - act )*np.log(1 - pred))))
#     # return -( np.mean ( (act * np.log(pred)) + ( ( 1 - act ) * np.log(1 - pred))))
# #  missing values


# a = np.array([1 , 2 , np.nan , 4 , np.nan , 6 , 7 , 8 , 9 , 10]) 
# print(a) ;
# print(np.isnan(a)) ;
# a = a[~np.isnan(a)]
# print(a) ;

import numpy as np
import matplotlib.pyplot as plt

x = np.linspace(-10, 10, 10, dtype=int)
y =  (x **4 +4*x**3 + 6*x**2 + 4*x + 1)
y = x* np.log(x)
plt.plot(x, y)
plt.show()