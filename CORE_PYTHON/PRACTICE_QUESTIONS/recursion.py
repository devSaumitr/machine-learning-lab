# RECURSION practice

# 1. countdown with base case
def countdown(n):
    if n <= 0:
        return
    print(n)
    countdown(n - 1)

countdown(5)

# 2. factorial
def factorial(n):
    if n == 1:
        return 1
    return n * factorial(n - 1)

print(factorial(5))

# 3. sum of a list
def list_sum(nums):
    if not nums:
        return 0
    return nums[0] + list_sum(nums[1:])

print(list_sum([1, 2, 3, 4]))

# 4. fibonacci (slow version - just for practice)
def fib(n):
    if n <= 1:
        return n
    return fib(n - 1) + fib(n - 2)

print(fib(6))

# 5. power
def power(base, exp):
    if exp == 0:
        return 1
    return base * power(base, exp - 1)

print(power(2, 3))

# 6. sum of digits
def digit_sum(n):
    if n < 10:
        return n
    return n % 10 + digit_sum(n // 10)

print(digit_sum(1234))

# 7. tree depth (ML style)
def tree_depth(node):
    if not node.get("children"):
        return 1
    return 1 + max(tree_depth(c) for c in node["children"])

tree = {"children": [{"children": [{}]}, {}]}
print(tree_depth(tree))

# --- PREDICT before running ---
# no base case -> RecursionError
# def f(n):
#     return n * f(n - 1)
# print(f(5))

# missing return -> None
def f(n):
    if n <= 0:
        return 0
    f(n - 1)

print(f(3))

# sum of odd numbers
def odd_sum(n):
    if n <= 0:
        return 0
    if n % 2 == 0:
        return odd_sum(n - 1)
    return n + odd_sum(n - 1)

print(odd_sum(5))

# --- try yourself ---
# reverse a string recursively
# check if a string is a palindrome recursively
# rewrite factorial with a loop and compare
# check recursion limit: import sys; print(sys.getrecursionlimit())