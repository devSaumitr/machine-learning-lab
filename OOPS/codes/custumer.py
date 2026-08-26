class Customer:

    def __init__(self, name, age, address):
        self.name = name
        self.age = age
        self.address = address

    def display_details(self):
        print(
            f"Name: {self.name}\n"
            f"Age: {self.age}\n"
            f"Address: {self.address.get_address()}"
        )

    def edit_profile(self, name, age, add):
        self.name = name
        self.age = age
        self.address.update_address(add)


class Address:

    def __init__(self, state, city, pincode):
        self.__city = city
        self.__state = state
        self.__pincode = pincode

    def get_state(self):
        return self.__state

    def get_city(self):
        return self.__city

    def get_pincode(self):
        return self.__pincode

    def get_address(self):
        return f"{self.__state}, {self.__city}, {self.__pincode}"
    def return_address(self):
            return [self.__state, self.__city, self.__pincode]
    def update_address(self, new_add):
         
        self.__state ,self.__city ,self.__pincode   =  new_add.return_address()
        

add = Address("Uttar Pradesh", "Lucknow", 226021)

new_add = Address("U.P.", "Gr. Noida", 201310)

cmt = Customer("Saumitra", 22, add)

cmt.display_details()

cmt.edit_profile("Roshan", 22, new_add)

cmt.display_details()