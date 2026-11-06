class BankAccount : 
    def __init__(self ):
        self._balance = 0 ;
    @property
    def balance(self) : 
         return self._balance 
    @balance.setter
    def balance  ( self , value) :
         if ( value  < 0 ) :
               print( "Balance cannot be negative");


         else :
              self._balance = value ;

    @property
    def double_balance (self) :
          return 2 * self._balance;


acc = BankAccount()

acc.balance = 500
print(acc.balance)

acc.balance = -100

print(acc.double_balance)