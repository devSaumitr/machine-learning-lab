class Atm:

    # Constructor
    def __init__(self):
        print(id(self))
        print("Hi, I am invoked")

        self.pin = ""
        self.__balance = 0

     #    self.menu()
     
    # Menu
    def menu(self):
        user_inp = int(input("""
        Hi, how can I help you?

        1: Press 1 to create PIN
        2: Press 2 to change PIN
        3: Check Balance
        4: Withdraw Money
        5: Press anything to exit

        """))

        if user_inp == 1:
            self.create_new_pin()

        elif user_inp == 2:
            self.update_new_pin()

        elif user_inp == 3:
            self.check_balance()

        elif user_inp == 4:
            self.withdraw()

        else:
            exit()

     #    self.menu()
    def set__balance(self):
        print("money deposited in the bank  "  ,self.__balance )
        
        self.__balance  += int(input("enter the amount to be depositied"))
       
        print("money deposited in the bank  "  ,self.__balance )
               
    
    
     #    self.menu()
    def get__balance(self):
        print("money deposited in the bank  "  ,self.__balance )
    
    
    # Create PIN
    def create_new_pin(self):
        print("Create new PIN")

        self.pin = int(input("Enter PIN: "))

        print("PIN created successfully")

        print("Enter bank balance")
        self.__balance = int(input("Enter balance: "))

        self.menu()

    # Change PIN
    def update_new_pin(self):
        print("Update PIN")

        o_pin = int(input("Enter existing PIN: "))

        if o_pin == self.pin:
            print("You can update PIN")

            self.pin = int(input("Enter new PIN: "))

            print("PIN updated successfully")

            self.menu()

        else:
            print("Wrong PIN")

            self.menu()

    # Check Balance
    def check_balance(self):
        o_pin = int(input("Enter existing PIN: "))

        if o_pin == self.pin:
            print("Your current balance is:", self.__balance)

            self.menu()

        else:
            print("Wrong PIN")

            self.menu()

    # Withdraw Money
    def withdraw(self):
        o_pin = int(input("Enter existing PIN: "))

        if o_pin == self.pin:

            amt = int(input("Enter the amount to withdraw: "))

            if amt <= self.__balance:

                print(f"{amt} is withdrawn successfully")

                self.__balance -= amt

                print("Your current balance is:", self.__balance)

                self.menu()

            else:
                print("Insufficient balance")

                self.menu()

        else:
            print("Wrong PIN")

            self.menu()


# Creating object
my = Atm ()
# print(type(my))
print(my)
my.menu()

my.__balance = 5000 ;
