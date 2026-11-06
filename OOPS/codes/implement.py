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
    
print(Student._Student__unique_id)