# __name__ and __main__ practice

# run this file directly and see:
print(__name__)

# the guard pattern
if __name__ == "__main__":
    print("this file is being RUN DIRECTLY")

# guard only controls its own block
if __name__ == "__main__":
    print("A")
print("B")

# main() pattern
def main():
    print("main program logic")
    for i in range(3):
        print(i)

if __name__ == "__main__":
    main()

# --- PREDICT before running ---
# 1. run this file directly:
#    what prints first?  (__main__)
# 2. now create mymod.py with just:  print(__name__)
#    and import it from here:        import mymod
#    what does mymod print?          (mymod)

# --- try yourself ---
# create mymod.py:
#   print("inside mymod, __name__ =", __name__)
# run it directly, then import it - observe the two values
# put a training-style function behind the guard and import it -
#   verify it does NOT run
# remove the guard and import again - watch side effects run