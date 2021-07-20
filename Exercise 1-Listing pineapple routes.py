import math
import random
import numpy as np
import io
from io import StringIO
portnames = ["PAN", "AMS", "CAS", "NYC", "HEL"]
 
def permutations(route, ports):
    # write the recursive function here
    for i in range(len(ports)):
        route.append(ports[i])
        permutations(route, ports[:i] + ports[i+1:])
        route.pop()
        # remember to print out the route as the recursion ends
    if len(ports) == 0:
        print(' '.join([portnames[i] for i in route]))

# this will start the recursion with 0 as the first stop
permutations([0], list(range(1, len(portnames))))