# =============================================
# MAP / FILTER / REDUCE - practice + technical
# =============================================

from functools import reduce

# ---------- PART 1: MAP (apply function to every item) ----------

numbers = [1, 2, 3, 4]

# loop version (the long way)
doubled = []
for n in numbers:
    doubled.append(n * 2)
print(doubled)

# map version
doubled = list(map(lambda x: x * 2, numbers))
print(doubled)

# map with two collections - pairs them up
a = [1, 2, 3]
b = [10, 20, 30]
print(list(map(lambda x, y: x + y, a, b)))  # [11, 22, 33]

# map without lambda - pass any function
words = ["apple", "kiwi", "fig"]
print(list(map(len, words)))                    # [5, 4, 3]
print(list(map(str.upper, ["hi", "there"])))    # ['HI', 'THERE']

# ML style: convert string numbers to float
raw = ["1.5", "2.5", "4.0"]
print(list(map(float, raw)))  # [1.5, 2.5, 4.0]

# ---------- PART 2: FILTER (keep items that pass a test) ----------

evens = list(filter(lambda x: x % 2 == 0, [1, 2, 3, 4, 5, 6]))
print(evens)  # [2, 4, 6]

big = list(filter(lambda x: x > 2, [1, 2, 3, 4]))
print(big)  # [3, 4]

# filter uses TRUTHINESS, not exact True/False
words = ["", "hello", "", "world", " "]
print(list(filter(None, words)))  # ['hello', 'world', ' '] - empties dropped

print(list(filter(len, ["cat", "", "dog", ""])))  # ['cat', 'dog']

# ML style: filter bad samples
rows = [{"name": "a", "age": 25}, {"name": "b", "age": -5}, {"name": "c", "age": 18}]
valid = list(filter(lambda r: r["age"] >= 0, rows))
print(valid)

# ---------- PART 3: REDUCE (combine all into one value) ----------

# sum
total = reduce(lambda a, b: a + b, [1, 2, 3, 4])
print(total)  # 10

# product
product = reduce(lambda a, b: a * b, [1, 2, 3, 4])
print(product)  # 24

# biggest
biggest = reduce(lambda a, b: a if a > b else b, [4, 9, 2, 7])
print(biggest)  # 9

# with initial value - first a becomes the initial value
print(reduce(lambda a, b: a + b, [1, 2, 3], 100))  # 106

# strings can be combined too
print(reduce(lambda a, b: a + b, ["a", "b", "c"]))  # 'abc'

# ---------- PART 4: PREDICT drills (say answer first) ----------

# 1. predict: squares
print(list(map(lambda x: x ** 2, [1, 2, 3])))  # [1, 4, 9]

# 2. predict: trap - lazy! map object, NOT a list
print(map(lambda x: x * 2, [1, 2, 3]))  # <map object at 0x...>

# 3. predict: filter keeps items, map returns results
print(list(filter(lambda x: x > 2, [1, 2, 3])))  # [3]
print(list(map(lambda x: x > 2, [1, 2, 3])))     # [False, False, True]

# 4. predict: map pairs items from both lists
print(list(map(lambda x, y: x * y, [1, 2, 3], [10, 10, 10])))  # [10, 20, 30]

# 5. predict: chaining - filter first, then map
nums = [1, 2, 3, 4, 5]
result = list(map(lambda x: x * 2, filter(lambda x: x % 2 == 0, nums)))
print(result)  # [4, 8]

# 6. predict: reduce trace
# reduce(lambda a, b: a + b, [1, 2, 3, 4])
# step 1: a=1, b=2 -> 3
# step 2: a=3, b=3 -> 6
# step 3: a=6, b=4 -> 10
print(reduce(lambda a, b: a + b, [1, 2, 3, 4]))  # 10

# ---------- PART 5: TECHNICAL SECTION ----------
# map/filter return ITERATORS (lazy) - nothing is computed until consumed
#   -> that's why you must wrap in list() to see values
#   -> same laziness as range, generators (see generators.py)
#   -> they are single-use: once consumed, gone
m = map(lambda x: x * 2, [1, 2, 3])
print(list(m))  # [2, 4, 6]
print(list(m))  # [] - consumed!

# map vs filter - the key difference:
#   map    -> keeps the RESULT of the function (same size as input)
#   filter -> keeps the ITEM, only if the function is truthy (smaller or same)

# reduce trace (accumulator pattern):
#   reduce(f, [a, b, c])  =  f(f(a, b), c)
#   the previous result becomes the next a - like a running total

# reduce is NOT built-in - it lives in functools
#   because most reduce uses have clearer built-ins:
#   reduce(lambda a, b: a + b, nums)  ==  sum(nums)
#   reduce(lambda a, b: max(a, b), nums)  ==  max(nums)

# comprehension equivalents (often more readable):
#   list(map(lambda x: x * 2, nums))  ==  [x * 2 for x in nums]
#   list(filter(lambda x: x % 2 == 0, nums))  ==  [x for x in nums if x % 2 == 0]

# ---------- PART 6: CHALLENGES (write your own) ----------
# 1. map: convert ["1.5", "2.5", "abc", "4.0"] - keep only valid floats,
#    then convert (combine filter + map)
# 2. filter a list of dicts: keep rows where 18 <= age <= 60
# 3. reduce: find the longest string in a list
# 4. reduce: count occurrences of a letter across a list of words
# 5. convert all map/filter above to comprehensions and verify same result