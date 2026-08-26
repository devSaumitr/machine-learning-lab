# MODULES & PACKAGES practice

# this file practices the IMPORT forms
# you need helpers.py in the same folder - create it if missing

# Form 1 - import the whole module
import helpers

print(helpers.double(4))
print(helpers.greet("ram"))

# Form 2 - import specific names
from helpers import double, greet

print(double(3))
print(greet("shyam"))

# Form 3 - alias
import helpers as h

print(h.double(5))

# --- PREDICT before running ---
# the import process: search -> execute -> cache

# a module is an object
import math
print(type(math))
print(math.floor(3.7))

# aliases point to the same cached module
import numpy as np
import numpy as n
print(np is n)

# check what has been imported
import sys
print("math" in sys.modules)
print("helpers" in sys.modules)

# --- try yourself ---
# create helpers.py with double() and greet() functions
#   (or a square() and cube() function)
# import helpers twice - notice the top-level print runs only once
# check sys.path - find where numpy is installed
# create a package folder with __init__.py and import from it