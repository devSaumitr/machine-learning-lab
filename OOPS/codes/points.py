class points :
    def __init__(self , x, y):
        self.x = x 
        self.y = y 

    
    def __str__(self) :
        return f"<{self.x},{self.y}>"
    
    
    def eucladian_dist(self, other ):
            return ((self.x - other.x) ** 2  + (self.y -other.y) ** 2 ) **  1/2
    

    def dist_from_origin(self):
        return (self.x** 2 + self.y**2 ) ** .5 ;
    
p1 = points(3,4)
p2 = points(1,1)
print(p1.eucladian_dist(p2))    

print(p1.dist_from_origin())
    
    
    
class Line :
    def __init__ (self ,A, B,C) :
        self.A = A    
        self.B = B
        self.C = C 
        
    
    
    def __str__(self) :
         return f"{self.A}x+ {self.B}y + {self.C} = 0"
              
              
              
    def point_on_line( line , point):
        
        if(line.A * point.x + line.B *point.y + line.C == 0 ):
            return f"{point.x },{point.y} lies on the line "                
        else :
            return f"{point.x },{point.y}  does not lie on the line "
    
    
    def shortest_dis_bw_line_and_a_point(line ,point):
        val = line.A * point.x + line.B*point.y+ line.C
        val = val / (line.A**2 +line.B**2 )**0.5
        return val 
l1 = Line(1,1,-2)
print(l1.point_on_line(p1)        )
print(l1.shortest_dis_bw_line_and_a_point(p1))