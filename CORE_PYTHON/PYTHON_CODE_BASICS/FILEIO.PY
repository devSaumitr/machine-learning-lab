# we have to open a file

# f = open("tt.txt" , "r") ; 
# data = f.readline() ;
# for char in data :
#     print (char)
# f.close()


# f= open("tt.txt" , "w")
# f.write( " hi its good to see this")
# f.close()

# print(f.read() );
# with open("tt.txt", "r") as f:
#     print(f.read(2))
#     print(f.read(2))
#     print(f.read(2))
#     print(f.tell())
import os 
with open ("tt.txt" , "w+") as f :
    print(f.read())
    print(f.tell())
    f.write(" this is something new")
    print(f.tell())
    f.seek(0)
    print(f.tell())
    print(f.read())
    print(f.tell())
    f.seek(0)
    print(f.tell())
    f.write (" JAVA")
    print(f.tell())
    f.seek(0)
    print(f.tell())
    print(f.read());
    print(f.tell())



if (os.path.exists("ttt.txt")) : 
    print(" file existing")
    os.remove("ttt.txt");
else :
    print("file does not exist")