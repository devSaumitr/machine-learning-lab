with open("ttt.txt", "w+") as f:
    f.write("Hi everyone\n")
    f.write("we are learning file i/o\n")
    f.write("using java\n")
    f.write("I like programming in java\n")

    f.seek(0)

    data = f.read()
    print(data)

    data = data.replace("java", "python")

    f.seek(0)
    f.truncate()

    f.write(data)

    f.seek(0)
    print(f.read())
    f.seek(0); 


def word_check(w) : 
    with open("ttt.txt" , "r" ) as f : 
        data = f.read()
        if(data.count(w) == 0) : 
            print(w, " does not exist");
        else :
             print(w, "exists in file");




# word_check("learninng")

        
def first_line(w) : 
    with open ("ttt.txt " , "r" )  as f :
        data = f.readlines()
        count = 0 
        for line in data :
            if(line.count(w) != 0) : 
                return  count  + 1 
            else :
                count += 1  ;
    return  -1 ;       

print(first_line("learning"))


lst = data.split(",")