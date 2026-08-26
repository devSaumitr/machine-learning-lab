# class Banking:

#     def __init__(self, balance, account, password):
#         self.balance = balance
#         self.account = account
#         self.__password = password

#     def display_account(self):
#         print("User account :", self.account)

#     def balance_left(self):
#         print("Current balance :", self.balance)

#     def credit(self, amt):
#         self.balance += amt
#         print("Amount credited :", amt)
#         self.balance_left()

#     def debit(self, amt):
#         if amt > self.balance:
#             print("Insufficient balance!")
#         else:
#             self.balance -= amt
#             print("Amount debited :", amt)
#             self.balance_left()

#     def login(self):
#         pwd = input("Enter password : ")

#         if pwd == self.__password:
#             print("Login successful")
#         else:
#             print("Wrong password")

#     def reset_pwd(self):
#         new_pwd = input("Enter new password : ")
#         self.__password = new_pwd
#         print("Password reset successful")

#     def password_hint(self):
#         if len(self.__password) >= 2:
#             print(
#                 "Password hint :",
#                 self.__password[0],
#                 "....",
#                 self.__password[-1]
#             )
#         else:
#             print("Password is too short for a hint")


# # Account 1
# b1 = Banking(10000, "1xpr", "abcde")

# print("\n===== ACCOUNT 1 =====")
# b1.display_account()
# b1.credit(1500)
# b1.debit(2000)
# b1.login()
# b1.reset_pwd()
# b1.password_hint()

# # Account 2
# b2 = Banking(21323, "1xrrr", "mypassword")

# print("\n===== ACCOUNT 2 =====")
# b2.display_account()
# b2.credit(1500)
# b2.debit(50000)  # Will show insufficient balance
# b2.login()
# b2.reset_pwd()
# b2.password_hint()



class person :
     def __init__(self , name , age):
        self.name = name 
        self.__age =  age;
     def __hello(self) :
         print( " hi nice to meet you " , self.name);

     def agee(self) : 
         print(" age is "  , self.__age) ; 



p = person("rahul" , 23 )
(p._person__hello())
p.agee    ()      
          