# Python OOP — Private Variables Explained (Full Chat)

> Source: [ChatGPT shared chat](https://chatgpt.com/share/6a87df27-9fd4-83e8-8dca-e5ae3e234094)
> Topic: Private variables, name mangling, encapsulation, inheritance, MRO, `super()`, abstraction

---

## Q1. Explain private variables of a class in Python — their accessibility, encapsulation, and how they are stored in memory

### 1. First: what is a class variable / instance variable?

```python
class Student:
    def __init__(self, name, marks):
        self.name = name
        self.marks = marks

s1 = Student("Adarsh", 90)
```

When you do `s1.name`, you are accessing data belonging to the **object `s1`**.

Conceptually, memory looks something like:

```text
s1
 │
 ▼
┌─────────────────────┐
│ Student object      │
│                     │
│ name  ──────────────┼──► "Adarsh"
│ marks ──────────────┼──► 90
└─────────────────────┘
```

`self.name` and `self.marks` are **instance attributes**.

---

### 2. So what does "private variable" mean?

Python doesn't have truly private variables in the same strict sense as C++.
Instead, Python uses **name mangling** to make an attribute harder to access accidentally.

```python
class Student:
    def __init__(self):
        self.__marks = 90
```

You might expect `print(s.__marks)` to work. It doesn't — you get:

```text
AttributeError
```

So you might think *"Python has hidden the variable."* But that's not exactly what happened.

---

### 3. The important concept: name mangling

Python internally changes:

```python
self.__marks
```

into approximately:

```python
self._Student__marks
```

Therefore this actually works:

```python
s = Student()
print(s._Student__marks)
```

> **Python's `__variable` is not truly private. Python uses name mangling to make direct accidental access difficult.**

---

### 4. Why does Python do this?

Mainly to prevent **accidental interference**. The class controls how its data is accessed, rather than allowing every piece of code to freely manipulate `s.marks = -500`. This is related to **encapsulation**.

---

### 5. Encapsulation ("modulation")

> **Keep an object's data and the methods that operate on that data together, while controlling how outside code interacts with that data.**

```python
class BankAccount:
    def __init__(self, balance):
        self.__balance = balance

    def deposit(self, amount):
        if amount > 0:
            self.__balance += amount

    def get_balance(self):
        return self.__balance


account = BankAccount(1000)
account.deposit(500)
print(account.get_balance())
```

The class controls the operation.

---

### 6. Three levels you should know in Python

| Syntax | Meaning | Enforced? |
|---|---|---|
| `self.name` | Public — intended to be accessible normally | — |
| `self._name` | "Internal/protected use" — **convention only** | No, `s._name` still works |
| `self.__name` | Triggers **name mangling** → `_ClassName__name` | Partially — `s._ClassName__name` works |

---

### 7. Connect this to memory

```python
class Student:
    def __init__(self, name, marks):
        self.name = name
        self.__marks = marks

s = Student("Adarsh", 90)
print(s.__dict__)
```

Output:

```python
{'name': 'Adarsh', '_Student__marks': 90}
```

Notice: `__marks` is NOT stored literally as `__marks`. Because of name mangling, the attribute is stored as `_Student__marks`.

```text
Student object
        │
        ▼
┌─────────────────────────────┐
│ instance attribute storage  │
│                             │
│ "name"            → "Adarsh"│
│ "_Student__marks" → 90      │
└─────────────────────────────┘
```

---

### 8. Is the private variable stored somewhere different in memory?

**No.** This is a very common misconception. Python does **not** create a special secret memory section for `__marks`. It is still an attribute of the object — the difference is primarily the **name**.

> **Private ≠ separate memory.**
> **Private ≈ name mangling + convention/access control through the class interface.**

---

### 9. How does Python find it?

Inside `Student`, writing `self.__marks` effectively looks for `self._Student__marks`, because Python knows `__marks` belongs to the `Student` class context.

---

### 10. Why is this useful with inheritance? (Name collision prevention)

```python
class Parent:
    def __init__(self):
        self.__value = 10   # _Parent__value → 10

class Child(Parent):
    def __init__(self):
        super().__init__()
        self.__value = 20   # _Child__value → 20
```

The child doesn't accidentally overwrite the parent's internal attribute. This is **name collision prevention**.

---

### 11. Compare with C++

```cpp
class Student {
private:
    int marks;
public:
    int getMarks() { return marks; }
};
```

- **C++:** `private` → compiler enforces access restriction. `s.marks` is rejected by the compiler.
- **Python:** `__marks` → name mangling → `_Student__marks`. Technically `s._Student__marks` can access it.

So Python's privacy is **not a security boundary**.

---

### 12. The complete mental model

```text
                 Python object
                       │
                       ▼
              ┌─────────────────┐
              │    __dict__     │
              │                 │
              │ name → "Adarsh" │
              │                 │
              │ _Student__marks │
              │       → 90      │
              └─────────────────┘
                       ▲
                       │
        __marks ──name mangling──► _Student__marks
```

> **"This isn't a magical hidden variable. Python changes its attribute name to `_ClassName__marks`, making accidental outside access harder and preventing name collisions."**

---

## Q2. Does `my.__balance = 5000` create a new instance variable?

User's ATM class code (constructor, menu, create/update PIN, check balance, withdraw) with `self.pin` public and `self.__balance` private.

**Answer: Exactly — yes.** This is the key point about Python's `__private` variables.

Inside `__init__`:

```python
self.__balance = 0    # becomes self._Atm__balance = 0
```

Object conceptually:

```text
my
 ↓
┌─────────────────────────────┐
│ Atm object                  │
│                             │
│ pin → ""                    │
│ _Atm__balance → 0           │
└─────────────────────────────┘
```

Now you do:

```python
my.__balance = 5000
```

Because you're writing `__balance` **outside the class**, the name-mangling transformation does **not** happen. Python simply creates a completely new attribute literally called `__balance`:

```text
┌─────────────────────────────────┐
│ Atm object                      │
│                                 │
│ pin → ""                        │
│ _Atm__balance → 0   ← original │
│ __balance → 5000      ← NEW!   │
└─────────────────────────────────┘
```

> **`my.__balance = 5000` creates a new instance attribute. It does NOT modify the existing private `__balance`.**

Prove it:

```python
print(my.__dict__)
# {'pin': '', '_Atm__balance': 0, '__balance': 5000}
```

And here's the interesting part — if you then call `my.check_balance()`, the method contains `self.__balance` **inside the `Atm` class**, so Python interprets it as `self._Atm__balance` and prints `0`, **not 5000**.

Remember the distinction:

```text
Outside class:  my.__balance     → literally "__balance"
Inside class:   self.__balance   → self._Atm__balance
```

That's the whole trick behind Python's "private" variables.

---

## Q3. Internal attributes are also references to the data loaded into objects?

Yes — attributes stored on the object are basically **names/references associated with values/objects**.

```text
my
 ↓
┌─────────────────────────────┐
│       Atm object            │
│                             │
│ pin ────────────────┐       │
│                     ↓       │
│                  "" object  │
│                             │
│ _Atm__balance ──────┐       │
│                     ↓       │
│                    0 object │
└─────────────────────────────┘
```

If you later do `self.pin = 500`, you're changing what `pin` refers to:

```text
Before:  pin ──→ 100
After:   pin ──→ 500
```

> **Object attributes are names stored in the object's attribute namespace (`__dict__`) that reference values/objects.**

---

## Q4. Class attribute counter — what are my mistakes?

User's code had `def __init(...)` (missing trailing underscores) and used `student.__unique_id` as a shared counter.

**Mistake 1 — constructor spelling:**

```python
def __init(*self, name, roll):    # ❌
def __init__(self, name, roll):   # ✅ two underscores on BOTH sides
```

**Corrected code:**

```python
class Student:
    __unique_id = 101

    def __init__(self, name, roll):
        self.name = name
        self.roll = roll
        self.unique_id = Student.__unique_id
        Student.__unique_id = Student.__unique_id + 1


s4 = Student("RAT", 13)
s3 = Student("DEAN", 23)
s1 = Student("ewa", 33)
s2 = Student("mad", 43)

student_list = [s1, s2, s3, s4]
for i in student_list:
    print(i.unique_id)
```

Output:

```text
103
104
102
101
```

(Because creation order was s4→101, s3→102, s1→103, s2→104 but printing order is s1, s2, s3, s4.)

**The important concept — two different attribute types:**

- `__unique_id` (→ `_Student__unique_id`) is a **class attribute** — one shared counter belonging to the class:

```text
Student CLASS
       │
       └── _Student__unique_id → 101 → 102 → 103 ...
```

- `self.unique_id` is a separate **instance attribute** for each student object:

```text
s4 object
│
├── name → "RAT"
├── roll → 13
└── unique_id → 101
```

**Terminology correction:** don't say "`__unique_id` is the private variable of every object." Say:

> **`Student.__unique_id` is one class attribute shared by the class, used as a counter. `self.unique_id` is a separate instance attribute belonging to each student object.**

---

## Q5. Why can't I access `__unique_id` directly?

Because it has **two leading underscores**, Python name-mangles it to `_Student__unique_id`.

```python
print(Student.__unique_id)             # ❌ AttributeError
print(Student._Student__unique_id)     # ✅ works (mangled name)
```

```text
Inside Student class:   __unique_id  → _Student__unique_id
Outside the class:      Student.__unique_id → looks for literally "__unique_id" → ❌ doesn't exist
```

> `__variable` → Python changes its name to `_ClassName__variable`. It's **not truly inaccessible** — just name-mangled to discourage direct access and prevent accidental collisions.

---

## Q6. Customer + Address composition — print vs return bug

User's `get_address()` **printed** the address instead of **returning** it, so `display_details()` showed `Address : None` (a function without `return` automatically returns `None`).

**Fix — use `return`:**

```python
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


class Address:
    def __init__(self, state, city, pincode):
        self.__city = city
        self.__state = state
        self.__pincode = pincode

    def get_address(self):
        return f"{self.__state}, {self.__city}, {self.__pincode}"


add = Address("Uttar Pradesh", "Lucknow", 226021)
cmt = Customer("Saumitra", 22, add)
cmt.display_details()
```

Output:

```text
Name: Saumitra
Age: 22
Address: Uttar Pradesh, Lucknow, 226021
```

**The OOP concept happening here — composition / has-a relationship:**

```text
cmt (Customer object)
        │
        ├── name ──→ "Saumitra"
        ├── age ──→ 22
        └── address ─────────┐
                             ↓
                       Address object
                             │
                             ├── _Address__state → "Uttar Pradesh"
                             ├── _Address__city → "Lucknow"
                             └── _Address__pincode → 226021
```

A `Customer` **has an** `Address`. The `Customer` doesn't directly touch the private `__state/__city/__pincode` — it asks through `get_address()`. That's **encapsulation in action**.

---

## Q7. edit_profile bug — creating NEW attributes instead of updating existing ones

User wrote in `edit_profile()`:

```python
self.__name = Name          # ❌ NEW attribute (_Customer__name)
self.__age = age            # ❌ NEW attribute (_Customer__age)
self.__address.update_address(add)   # ❌ AttributeError — doesn't exist
```

But the actual attributes created in `__init__` were `self.name`, `self.age`, `self.address`. Writing `self.__name` inside the class creates **completely different** mangled names.

**Fix — modify the existing attributes:**

```python
def edit_profile(self, name, age, add):
    self.name = name
    self.age = age
    self.address.update_address(add)
```

Full corrected version:

```python
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

    def update_address(self, new_add):
        self.__state = new_add.get_state()
        self.__city = new_add.get_city()
        self.__pincode = new_add.get_pincode()


add = Address("Uttar Pradesh", "Lucknow", 226021)
new_add = Address("U.P.", "Gr. Noida", 201310)

cmt = Customer("Saumitra", 22, add)
cmt.display_details()
cmt.edit_profile("Roshan", 22, new_add)
cmt.display_details()
```

Output:

```text
Name: Saumitra
Age: 22
Address: Uttar Pradesh, Lucknow, 226021

Name: Roshan
Age: 22
Address: U.P., Gr. Noida, 201310
```

Notes:
- The `return self` inside `update_address()` isn't necessary unless chaining like `add.update_address(new_add)` is wanted.
- Conceptually: `Customer --has-a--> Address`. Encapsulation + composition working together.

---

## Q8. Inheritance — child checked first, then parent (MRO basics)

User's `User` / `Student(User)` example with inherited `login()`.

Key idea:

> **Inheritance in Python does not mean "parent object is created first and then child object." It means the child class can look for attributes/methods in its own class and, if not found, Python searches the parent according to the MRO.**

For `s.login()`:

```text
Student → login? ❌
User    → login? ✅  → use User.login()
```

If `Student` defines its own `login()`, lookup stops there (**method overriding**) — `User.login()` is NOT called first.

MRO can be seen with `print(Student.mro())`:

```text
Student → User → object
```

About the commented-out child `__init__`: if the child defines its own `__init__`, Python finds `Student.__init__` immediately and **does not automatically call `User.__init__`**. To get parent initialization, explicitly call:

```python
super().__init__(name, age)
```

Mental model:

> **Inheritance:** "If I ask a Student object for something, Python searches the Student class first, then follows the MRO to its parents until it finds it."
> **Constructor:** "`__init__` is just another method. If the child defines it, Python uses the child's version. Use `super()` when you want the parent initialization."

---

## Q9. Notes review — mutable defaults, inheritance rules, super(), types of inheritance, MRO

User shared revision notes; assistant verified/corrected them.

### 1. Mutable vs immutable containers

```python
t = (1, 2, 3)
t[0] = 10         # ❌ tuple immutable

t = ([1, 2], 3)
t[0] = [5, 6]     # ❌ cannot replace element
t[0].append(3)    # ✅ CAN modify the mutable list inside
```

> Immutable container → its elements/references cannot be replaced. But if an element itself is mutable, that object can still be modified.

### 2. Mutable default arguments

```python
def fun(x=[]):        # ❌ same list reused between calls
def fun(x=None):      # ✅
    if x is None:
        x = []
```

> Don't use a mutable object as a default argument unless you deliberately want shared state. Use `None` and create a new object inside.

### 3–5. Inheritance & constructors

```python
class Parent:
    def __init__(self):
        self.x = 10

class Child(Parent):
    def __init__(self):
        self.y = 20

c = Child()
c.y   # ✅
c.x   # ❌ AttributeError — Parent.__init__() never ran
```

> If the child defines its own `__init__`, Python uses it and does NOT automatically call the parent's `__init__`.

### 6. Private members of parent

Inside `Child`, `self.__x` mangles to `self._Child__x`, while the parent's is `self._Parent__x` — different names.

> A child class cannot directly access a parent's `__private` attribute using the same name. (Technically `self._Parent__x` would work — privacy is name mangling, not security.)

### 7. Method overriding

Child and parent have the same method → child's version found first → runs. Parent's method is overridden.

### 8–9. `super()`

```python
class Child(Parent):
    def __init__(self):
        super().__init__()
        self.y = 20
```

Correction: `super()` isn't "only for methods" — it accesses the **next class in the MRO** (commonly methods and constructors). It doesn't directly give parent instance attributes like `super().x`, because `x` lives on the **object**, not the parent class.

> `self` → object and its attributes. `super()` → access behavior from the next class in the MRO.

### 10. Parent reference cannot access child members

```python
p = Parent()
p.show_parent()  # ✅
p.show_child()   # ❌ — Parent object isn't a Child object
```

> Child object → can use Parent functionality. Parent object → cannot use Child functionality.

### 11. Types of inheritance

```text
Single:        Parent → Child
Multilevel:    Grandparent → Parent → Child
Hierarchical:  Parent → Child1, Child2
Multiple:      Parent1, Parent2, Parent3 → Child
```

### 12. Multiple inheritance + MRO

```python
class A:
    def show(self): print("A")

class B:
    def show(self): print("B")

class C(A, B):
    pass

c = C()
c.show()          # prints "A"
print(C.mro())    # C → A → B → object
```

Correction: the rule is not "the parent written earlier wins" — it's:

> **Python follows the MRO, and the first class in the MRO containing the requested method wins.**

### Final revision sheet

```text
MUTABLE / IMMUTABLE
────────────────────────────
Immutable container → cannot replace its elements
Mutable object inside → can still be modified

DEFAULT ARGUMENT
────────────────────────────
Avoid: def f(x=[])
Prefer: def f(x=None): if x is None: x = []

INHERITANCE
────────────────────────────
Child can use accessible parent functionality.
Method lookup: Child → Parent → ... → object

CONSTRUCTOR
────────────────────────────
Child has no __init__?  → Parent __init__ found and used.
Child has its own __init__? → Child's is used;
parent's is NOT automatically called.

super()
────────────────────────────
Accesses next class's functionality per MRO.
Common: super().__init__()

METHOD OVERRIDING
────────────────────────────
Same method in Child & Parent → Child's runs.

PRIVATE VARIABLE
────────────────────────────
__x → name mangling: _Parent__x vs _Child__x
Child can't access parent's __x by the same name.

OBJECT ACCESS
────────────────────────────
Child object → can access inherited parent members
Parent object → cannot access child members

MRO
────────────────────────────
C(A, B): C → A → B → object
```

> **Inheritance doesn't mean "parent runs first." It means Python has a class hierarchy to search when resolving methods and attributes.**

---

## Q10. More about `super()`

### 1. What problem does `super()` solve?

Without `super().__init__(name)`, `Student.__init__` replaces `User.__init__`, so `s.name` never gets created.

### 2. What `super()` actually means

> **`super()` gives you access to the next class in the MRO.**

For MRO `Student → User → object`, inside `Student`, `super()` means *"start looking after Student."*

### 3. `super()` is NOT the parent object

It's a special object that performs lookup according to the MRO.

### 4. Why not simply `User.__init__(self, name)`?

That works, but `super()` follows the **MRO**, which matters enormously with multiple inheritance.

### 5–7. Extending behavior

```python
class User:
    def login(self):
        print("User login")

class Student(User):
    def login(self):
        print("Student login")
        super().login()

s = Student()
s.login()
# Student login
# User login
```

> **Overriding changes the behavior. `super()` allows you to reuse the inherited behavior.**

### 8–9. `super()` chains & multiple inheritance

```python
class A:
    def show(self): print("A")

class B(A):
    def show(self):
        print("B"); super().show()

class C(B):
    def show(self):
        print("C"); super().show()

C().show()   # C, B, A  (MRO: C → B → A → object)
```

With `class C(A, B)` where both define `show()` and `C.show()` calls `super().show()`:

```text
MRO ≈ C → A → B → object
Output: C, A   ← super() went to A, NOT to "the other parent B"
```

### 10–12. Key subtleties

- Inside `A` (in MRO C→A→B), `super()` continues to **B** — powerful chain behavior.
- `super().name` is invalid — `name` is an attribute of the **object**; the parent constructor merely created it on the same object via `self`.
- Modern zero-argument `super()` automatically knows the current class and object (vs old `super(Student, self)`).

### ⭐ Final `super()` cheat sheet

```text
super()
│
├── Used inside inheritance
├── Gives access to the next class in MRO
├── Commonly used for parent constructor → super().__init__()
├── Can call other methods → super().method()
├── Not a parent object
├── Not limited to constructors
└── In multiple inheritance: follows MRO
```

> **`super()` does not mean "call my parent"; it means "continue the lookup from the next class in the MRO."**

---

## Q11. Quick confirmation of `super()` understanding

User summary confirmed with corrections:

1. `super()` is for MRO lookup ✅
2. Methods yes; instance data belongs to the **object** — `super().__init__()` creates `name` on the *same* Student object via `self` ✅
3. Overridden members can be accessed via `super()` ✅
4. Parent constructor accessed via `super()` operates on the same child object ✅

> **`super()` follows the MRO to find the next implementation of a method (including `__init__`); the method then operates on the same child object through `self`.**

---

## Q12. What is MRO in easy and short?

**MRO = Method Resolution Order**

> **The order in which Python searches classes to find a method or attribute.**

```python
class A:
    def show(self): print("A")
class B(A): pass
class C(B): pass

c.show()   # searches C → B → A, finds A.show()
```

Memory trick:

> **MRO = "Where will Python look first, second, third...?"** And `super()` uses this order to continue the lookup.

---

## Q13. Teach me abstraction in Python

### 1. What is abstraction?

> **Abstraction means hiding the unnecessary implementation details and showing only what the user needs to use.**

ATM example: you see `Check balance / Withdraw money / Change PIN` — you don't see how the bank server, database, authentication, or transaction processing works internally.

### 2. Real-life example

`print("Hello")` — you don't need to know how Python sends characters to the output system.

### 3. Abstraction in OOP

```python
class Car:
    def start(self):
        # complicated engine logic
        print("Car started")
```

User only needs `car.start()`. Implementation hidden behind the method.

### 4–6. Abstract classes & abstract methods

```python
from abc import ABC, abstractmethod

class Vehicle(ABC):
    @abstractmethod
    def start(self):
        pass

class Car(Vehicle):
    def start(self):
        print("Car starts with key")

class Bike(Vehicle):
    def start(self):
        print("Bike starts with self-start")
```

> An abstract method = **the parent says every child MUST provide its own implementation.**
> Parent defines **what must exist**; child defines **how it works**.

### 7. Why can't we instantiate `Vehicle()`?

Error — because it has an unimplemented abstract method. `Car()` works because `Car` implemented `start()`.

### 8. The contract idea

```text
             Vehicle
                │
        ┌───────┴───────┐
        ↓               ↓
       Car             Bike

Vehicle says: "Every child MUST have a start() method."
```

> **Abstract class = blueprint/contract for child classes.**

### 9. Abstraction vs Encapsulation

```text
Encapsulation → HOW to protect/hide data   (data protection)
Abstraction   → WHAT to show and hide      (implementation hiding)
```

### 10. Why use abstract classes? (Payment example)

```python
from abc import ABC, abstractmethod

class Payment(ABC):
    @abstractmethod
    def pay(self, amount):
        pass

class UPI(Payment):
    def pay(self, amount):
        print("Paid using UPI:", amount)

class CreditCard(Payment):
    def pay(self, amount):
        print("Paid using Credit Card:", amount)
```

System guarantee: whatever payment method it receives, it MUST have `pay()`.

### 11–12. Mental model

```text
                ABSTRACT CLASS
                      │
                      ↓
               Defines WHAT
               must be provided
                      │
               ┌──────┴──────┐
               ↓             ↓
              Car           Bike
               │             │
               ↓             ↓
         HOW it starts  HOW it starts
```

Final one-liner:

> **Abstraction in Python means defining a common interface/contract while hiding specific implementation details, often using abstract classes and `@abstractmethod`.**

Three things to remember:
- **`ABC`** → makes the class an abstract base class
- **`@abstractmethod`** → child must implement this method
- **Child class** → provides the actual implementation

---

## Q14. Java vs Python — abstraction, abstract class, functional interface

User explained their Java background (abstraction = remote control analogy; abstract class = may mix concrete + abstract methods; functional interface; can't instantiate abstract classes).

Corrections & mapping:

1. **Abstraction understanding correct** — know *what to use*, not *how it's implemented internally*.
2. **Vehicle example exactly right** — common interface `start()`, children implement their own logic.
3. **Abstract method remembered correctly** — signature given, body not defined.
4. **⚠ Functional interface correction:** a Java functional interface has **exactly ONE abstract method** (not "only abstract methods"). It may also have `default` and `static` methods. Useful with lambdas.
5. **Java ↔ Python mapping:**

```text
Java                         Python
abstract class               ABC
abstract method              @abstractmethod
extends                      inheritance
@Override                    method overriding
```

6. **Cannot instantiate abstract class** — same in Python. If child doesn't implement all abstract methods, child is still abstract and instantiation fails too.
7. **Overriding + abstraction work together:**

```text
ABSTRACTION: Vehicle says "every vehicle must have start()"
OVERRIDING:  Car says "here is MY implementation of start()"
```

8. **Polymorphism connection:**

```python
def start_vehicle(vehicle):
    vehicle.start()

start_vehicle(Car())    # Car starting
start_vehicle(Bike())   # Bike starting
```

The function only knows: *"this thing has a `start()` operation."*

9. **The four OOP pillars:**

```text
1. ENCAPSULATION → bundle data + methods; control access to internal data
2. ABSTRACTION   → hide implementation details; show only necessary interface
3. INHERITANCE   → child gets accessible functionality from parent
4. POLYMORPHISM  → same call → different behavior depending on object
```

10. **Terminology fix:** "abstract method is a method whose body was defined" is backwards —

```text
abstract method → WHAT (signature defined, body NOT provided)
concrete method → HOW  (signature + body provided)
```

---

## Q15. Can an abstract class have more than one abstract method?

**Yes, absolutely.**

```python
from abc import ABC, abstractmethod

class Vehicle(ABC):
    @abstractmethod
    def start(self): pass

    @abstractmethod
    def stop(self): pass

    @abstractmethod
    def fuel_type(self): pass
```

A child must implement **all of them** to become concrete:

```python
class Car(Vehicle):
    def start(self): print("Car starts")
    def stop(self): print("Car stops")
    def fuel_type(self): print("Petrol")

c = Car()   # ✅
```

Implement only two? Then `Car()` ❌ errors because `fuel_type()` is still abstract.

Important distinction:

- **Abstract class** → any number of abstract methods
- **Java functional interface** → exactly ONE abstract method
- **Python ABC** → any number of `@abstractmethod`s

> One abstract class → 1, 2, 10, or 100 abstract methods is completely fine.
