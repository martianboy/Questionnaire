import numpy as np
from colorImage import searchForColorPoints
from shape import shapesHaveOverlap, containsAnotherPoint

def FindPapers(im):
    criteria = [[(90, 160), (26, 100), (38, 100)],
                [(185, 230), (35, 100), (35, 85)],
                [(6, 30), (45, 100), (50, 100)],
                [(245, 330), (25, 65), (30, 80)]]

    centroids = searchForColorPoints(im, criteria)

    estimatedNumberOfPapers = min(list(len(x) for x in centroids))
    papers = []

    for p11 in centroids[0]:
        for p12 in centroids[1]:
            for p22 in centroids[2]:
                for p21 in centroids[3]:
                    if isPaper((p11,p12,p22,p21), np.concatenate(centroids)):
                        papers.append((p11,p12,p22,p21))

    papers = np.array(papers)
    
    # Pruning stage
    collisionGraph = np.zeros((len(papers), len(papers)), bool)
    
    for i,paper1 in enumerate(papers):
        for j,paper2 in enumerate(papers):
            collisionGraph[i,j] = shapesHaveOverlap(paper1, paper2)

    ps = np.array(range(len(papers)), np.uint8)
    for p in ps:
        ret = SelectPapers(ps, p, estimatedNumberOfPapers - 1, collisionGraph)
        if ret != None:
            ret.insert(0,p)
            return papers[ret]
    
def SelectPapers(papers, p, e, collision):
    b = collision[p,papers]
    new_set = papers[b.argsort()[:len(b)-sum(b)]]

    if len(new_set) < e:
        return None
    if e == 1 and len(new_set) == 1:
        return new_set.tolist()

    for new_point in new_set:
        ret = SelectPapers(new_set, new_point, e - 1, collision)
        if ret != None:
            ret.insert(0,new_point)
            return ret

    return None

def isPaper(points, allPoints):
    topEdge = points[1] - points[0]
    rightEdge = points[2] - points[1]
    bottomEdge = points[3] - points[2]
    leftEdge = points[0] - points[3]
    
    axiom1 = np.cross(topEdge, rightEdge) > 0
    axiom2 = np.cross(rightEdge, bottomEdge) > 0
    axiom3 = np.cross(bottomEdge, leftEdge) > 0
    axiom4 = np.cross(leftEdge, topEdge) > 0

    axiom5 = not containsAnotherPoint(points, allPoints)

    return axiom1 and axiom2 and axiom3 and axiom4 and axiom5
