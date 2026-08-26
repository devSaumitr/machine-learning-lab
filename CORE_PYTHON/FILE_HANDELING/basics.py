# First, read the existing data
with open("test.txt", "r") as f:
    d = f.read()

print("Original data:")
print(d)


# Store the same data twice
l = [d for i in range(2)]


# Open file for both writing and reading
with open("test.txt", "w+") as f:

    # Write the data twice
    f.writelines(l)

    # Pointer is currently at the end of the file
    # Move it back to the beginning
    f.seek(0)

    # Now read the updated file
    d = f.read()


print("Updated data:")
print(d)