from numpy import sign,cross

def containsAnotherPoint(paper, allPoints):
    topEdge = paper[1] - paper[0]
    rightEdge = paper[2] - paper[1]
    bottomEdge = paper[3] - paper[2]
    leftEdge = paper[0] - paper[3]
    
    for p in allPoints:
        a1 = cross(topEdge, p - paper[0]) > 0
        a2 = cross(rightEdge, p - paper[1]) > 0
        a3 = cross(bottomEdge, p - paper[2]) > 0
        a4 = cross(leftEdge, p - paper[3]) > 0

        if a1 and a2 and a3 and a4: return True

    return False

def shapesHaveOverlap(shape1, shape2):
    shapeEdges1 = list((shape1[(i+1) % 4], shape1[i]) for i in range(4))
    shapeEdges2 = list((shape2[(i+1) % 4], shape2[i]) for i in range(4))
    
    for line1 in shapeEdges1:
        for line2 in shapeEdges2:
            if lineSegmentsCollide(line1,line2):
                return True

    return False

def lineSegmentsCollide(line1, line2):
    u = line1[1] - line1[0]
    u1 = line2[1] - line1[0]
    u2 = line2[0] - line1[0]
    
    v = line2[1] - line2[0]
    v1 = line1[1] - line2[0]
    v2 = line1[0] - line2[0]

    a1 = sign(cross(v,v1)) != sign(cross(v,v2))
    a2 = sign(cross(u,u1)) != sign(cross(u,u2))

    return a1 and a2
