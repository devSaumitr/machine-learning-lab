class MyException(Exception):
     def __init__(self , message):
         print(message)



class ATM :
    def __init__(self, balance) : 
        self.balance = balance 
        
    def withdraw(self ,amount):
        if( amount < 0 ):
            raise MyException('amount can not be negative')
        if( self.balance < amount):
            raise MyException("you dont have enough money")
        self.balance -= amount
        print(f"transaction succesful --> you successfully made a transaction of {amount}")
        
b = ATM(50000) ;


try:
    b.withdraw(40000)        
except MyException as e :
    print(e)
    print("transaction unsuccesful")
else :
    print(f"  your current balance is {b.balance} " )
finally : 
    print("you tried to make a transaction")            
             