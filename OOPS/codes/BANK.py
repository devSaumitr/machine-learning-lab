from abc import ABC , abstractmethod


class BankApp(ABC) :
    def database (self):
        print ("connected to Database")
        
    
    
    @abstractmethod
    def security(self):
        pass        



class MobileApp(BankApp) :
    
    
    def mobile_login(self):
        
        print("mobile logged in ")
        
        
    
    def security() :
        print("security added")
        
        
            
            


m = MobileApp()
m.mobile_login() ;            