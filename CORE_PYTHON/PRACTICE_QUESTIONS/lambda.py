# =============================================
# LAMBDA FUNCTIONS - practice + technical
# =============================================

# ---------- PART 1: BASICS (ready to run) ----------

# lambda = one-line anonymous function
f = lambda x: x * 2
print(f(5))

# multiple params
g = lambda x, y: x + y
print(g(3, 4))

# no params
h = lambda: "hello"
print(h())

# default arguments work like def
add = lambda a=1, b=2: a + b
print(add())
print(add(b=10))

# conditional expression is allowed (still one expression)
grade = lambda score: "pass" if score >= 40 else "fail"
print(grade(60))
print(grade(30))

# *args works too
k = lambda *args: sum(args)
print(k(1, 2, 3))

# ---------- PART 2: REAL USES ----------

# sort with key
students = [("ram", 22), ("shyam", 19), ("karan", 24)]
by_age = sorted(students, key=lambda s: s[1])
print(by_age)

# case-insensitive sort
names = ["Aarav", "isha", "Vikram", "devan"]
print(sorted(names))
print(sorted(names, key=lambda n: n.lower()))

# max by some value (ML: best model)
models = [("rf", 0.91), ("xgb", 0.94), ("lr", 0.87)]
best = max(models, key=lambda m: m[1])
print(best)

# closure - lambda remembers its factor
def multiplier(factor):
    return lambda x: x * factor

double = multiplier(2)
triple = multiplier(3)
print(double(10))
print(triple(10))

# lambda with map / filter / sorted
words = ["apple", "kiwi", "fig"]
print(list(map(lambda w: w.upper(), words)))
print(list(filter(lambda w: len(w) > 3, words)))

# ---------- PART 3: PREDICT drills (say answer first) ----------

# 1. predict: 2 ** 5
print(lambda x, y: x ** y)  # -> <function <lambda> at ...> (function object)

f1 = lambda x, y: x ** y
print(f1(2, 5))  # 32

# 2. predict: sorted by string at index 1
pairs = [(3, "c"), (1, "a"), (2, "b")]
print(sorted(pairs, key=lambda p: p[1]))  # [(1, 'a'), (2, 'b'), (3, 'c')]

# 3. predict: print returns None, lambda returns that None
r = lambda: print("hi")
result = r()
print(result)  # hi then None

# 4. predict: late binding trap -> [2, 2, 2] not [0, 1, 2]
funcs = [lambda: i for i in range(3)]
print([f() for f in funcs])

# 5. predict: the fix -> [0, 1, 2]
funcs = [lambda i=i: i for i in range(3)]
print([f() for f in funcs])

# 6. predict: conditional inside lambda
f3 = lambda x: x * 2 if x > 0 else 0
print(f3(-3))  # 0
print(f3(5))   # 10

# ---------- PART 4: TECHNICAL SECTION ----------
# what lambda really is:
#   - lambda is an EXPRESSION, def is a STATEMENT
#   - both create the same kind of object: <class 'function'>
#   - lambda auto-returns its single expression - no return keyword
#   - limitations: one expression only, no statements (if/for/while),
#     no docstring, __name__ is always '<lambda>'

print(type(lambda x: x))            # <class 'function'>
print((lambda x: x).__name__)       # <lambda>

# late binding explained:
#   the loop variable i is ONE variable, shared by all lambdas
#   when the loop ends, i == 2 -> all lambdas see 2
#   lambda i=i: i copies the VALUE into a default argument at creation time

# LEGB: lambda looks up names like def (Local, Enclosing, Global, Built-in)
x = 10
print((lambda: x)())                # 10 - sees global x

# when NOT to use lambda:
#   - multiple statements needed -> use def
#   - logic is reused or non-trivial -> def (a name is documentation)
#   - readability matters -> def

# ---------- PART 5: CHALLENGES (write your own) ----------
# 1. lambda that returns absolute difference of two numbers
# 2. lambda that returns the last character of a string
# 3. sort [("apple", 5), ("kiwi", 2), ("fig", 1)] by word length
# 4. use max() with key to find the longest word in a list
# 5. write multiplier(factor) and verify each remembers its own factor