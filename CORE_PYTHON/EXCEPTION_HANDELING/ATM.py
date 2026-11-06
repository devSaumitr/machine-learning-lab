
class ATM :
    def __init__(self, balance) : 
        self.balance = balance 
        
    def withdraw(self ,amount):
        if( amount < 0 ):
            raise Exception('amount can not be negative')
        if( self.balance < amount):
            raise Exception("you dont have enough money")
        self.balance -= amount
        print("transaction succesful")
        
b = ATM(5000) ;


try:
    b.withdraw(40000)        
except Exception as e :
    print(e)
    print("transaction unsuccesful")
else :
    print(f"  your current balance is {b.balance} " )
finally : 
    print("you tried to make a transaction")            
    