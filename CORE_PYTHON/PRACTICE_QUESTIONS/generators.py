# GENERATORS practice

# 1. basic generator with yield
def give_numbers():
    yield 1
    yield 2
    yield 3

g = give_numbers()
print(g)
print(next(g))
print(next(g))
print(next(g))

# 2. pause and resume
def counter():
    print("starting...")
    x = 1
    yield x
    x += 1
    yield x
    x += 1
    yield x

c = counter()
print(next(c))
print(next(c))
print(next(c))

# 3. generator is an iterator - single use
def first_three():
    yield 10
    yield 20
    yield 30

print(list(first_three()))
print(sum(first_three()))

# 4. generator expression
squares_list = [x * x for x in range(10)]
squares_gen = (x * x for x in range(10))
print(type(squares_list))
print(type(squares_gen))

# 5. infinite stream
def evens():
    x = 0
    while True:
        yield x
        x += 2

e = evens()
print([next(e) for _ in range(5)])

# 6. generator with condition
g2 = (x for x in range(10) if x % 2 == 0)
print(list(g2))

# 7. while loop state persists
def gen():
    x = 0
    while x < 3:
        yield x
        x += 1

g3 = gen()
print(next(g3))
print(next(g3))

# 8. memory-friendly version of squares
def squares(n):
    for i in range(n):
        yield i * i

print(list(squares(5)))

# --- PREDICT before running ---
# trap: single use
g4 = (x for x in range(5))
print(sum(g4))
print(sum(g4))

# trap: return ends the generator
def g5():
    yield 1
    return
    yield 2

print(list(g5()))

# trap: next then list
g6 = (x for x in range(3))
print(next(g6))
print(list(g6))

# --- try yourself ---
# write a generator that yields squares of odd numbers only
# write a generator that yields words from a string one by one
# chain two generators (one feeds the other)
# try len() on a generator - observe the error