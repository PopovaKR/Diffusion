import gmshparser
import openmesh as om
import numpy as np
from scipy.integrate import tplquad, dblquad

import numpy as np
from scipy.integrate import tplquad, dblquad
import time

def time_of_function(function):
    def wrapped(*args):
        start_time = time.perf_counter_ns()
        res = function(*args)
        print(time.perf_counter_ns() - start_time)
        return res
    return wrapped

def net_init(filename):
    node_res = []
    tetra_res = []
    mesh = gmshparser.parse(filename)
    for entity in mesh.get_node_entities():
        for node in entity.get_nodes():
            ncoords = node.get_coordinates()
            node_res.append(ncoords)
    for entity in mesh.get_element_entities():
        for element in entity.get_elements():
            elcord = element.get_connectivity()
            tetra_res.append(elcord)
    # print("net_init")

    return [node_res, tetra_res]

def save_net(results):
    mesh = om.PolyMesh()
    vh = []
    fs = []
    for i in range(0, len(results[0])):
        vh1 = mesh.add_vertex(results[0][i])
        vh.append(vh1)

    for i in range(0, len(results[1])):
        face1 = mesh.add_face(vh[results[1][i][0]-1], vh[results[1][i][1]-1], vh[results[1][i][2]-1], vh[results[1][i][3]-1])
        fs.append(face1)
    om.write_mesh('result.obj', mesh)

# расчёт тензора D
def D(x):

    return [[[1 + x**2, 0, 0], [0, 0.1 * (1 + x**2), 0],[0, 0, 0.01 * (1 + x**2)]]] 

def volume_integral(f, tetrahedron):
    p0 = np.array(tetrahedron[0]).reshape((3, 1)) 
    p1 = np.array(tetrahedron[1]).reshape((3, 1))
    p2 = np.array(tetrahedron[2]).reshape((3, 1))
    p3 = np.array(tetrahedron[3]).reshape((3, 1))

    matrix = p1 - p0
    matrix = np.hstack((matrix, p2-p0))
    matrix = np.hstack((matrix, p3-p0))

    A = np.linalg.inv(matrix)
    jacob = np.linalg.det(A)

    def f_new(x, y, z):
        vec = np.array([x, y, z]).reshape((3, 1))
        vec_new = A@(vec - p0)
        return f(vec_new[0], vec_new[1], vec_new[2])/(jacob)

    def surf1(x, y):
        return 0

    def surf2(x, y):

        if 0 <= x <= 1 and 0 <= y <= 1:
            return 1 - x - y
        return 0

    def func1(x):
        return 0

    def func2(x):
        if 0 <= x <= 1:
            return 1-x
        return 0
    # print("volume_integral")

    return tplquad(f_new, 0, 1, func1, func2, surf1, surf2)[0]

def surface_integral(f, triangle):
    p0 = np.array(triangle[0]).reshape((2, 1))
    p1 = np.array(triangle[1]).reshape((2, 1))
    p2 = np.array(triangle[2]).reshape((2, 1))

    matrix = p1 - p0
    matrix = np.hstack((matrix, p2 - p0))
    A = np.linalg.inv(matrix)
    jacob = np.linalg.det(A)

    def f_new(x, y):
        vec = np.array([x, y]).reshape((2, 1))
        vec_new = A@(vec - p0)
        return f(vec_new[0], vec_new[1])/(jacob)

    def func1(x):
        return 0

    def func2(x):
        if 0 <= x <= 1:
            return 1-x
        return 0
    # print("surface_integral")

    return dblquad(f_new, 0, 1, func1, func2)[0]

def phi(x:np.array, p1:np.array, p2:np.array, p3:np.array, p4:np.array):
    P = np.array([p1, p2, p3, p4]).transpose()
    P = np.vstack((P, [1, 1, 1, 1]))
    P = np.linalg.inv(P)
    x = np.append(x, 1)
    ans = P @ x
        
    # print("phi")
    return ans

def grad_phi(p1:np.array, p2:np.array, p3:np.array, p4:np.array):
    P = np.array([p1, p2, p3, p4]).transpose()
    P = np.vstack((P, [1, 1, 1, 1]))
    P = np.linalg.inv(P)
    # print("grad_phi")  
    return np.array([P[0, 0:3],P[1, 0:3], P[2, 0:3], P[3, 0:3]])


def right_side(x:np.array):
#Функция, возвращающая правую часть r(x) уравнения
    r = - (0.05*x[0]+0.0006875*x[0]**2+0.0006875)*np.exp((x[0]+x[1]+20)/40) - 0.001*(x[0]**2+1)
    # print("right_side")
    return(r)


def dirichlet(x:np.array):
#Функция, возвращающая граничное условие Дирихле
    gD = np.exp((x[0]+x[1]+20)/40) + (x[2]+10)**2/20
    # print("gD")
    return gD

def neumann(x:np.array,p0:np.array, p1:np.array, p2:np.array):
#Функция, возвращающая граничное условие Неймана
    dgradu = np.array([0.025*(1+x[0]**2)*np.exp((x[0]+x[1]+20)/40), 0.0025*(1+x[0]**2)*np.exp((x[0]+x[1]+20)/40), 0.001*(x[0]**2+1)*(x[2]+10)])
    norma = np.ones(3)
    vec1 = p1 - p0
    vec2 = p2 - p0 
    norma = np.cross(vec1, vec2)
    norma = norma / np.linalg.norm(norma)
    gN = np.matmul(norma, dgradu)
    # print("neumann")
    return gN

def solver(net):

    Nodes = net[0]
    N = len(Nodes)

    C = net[1]

    A = np.zeros((N, N))

    R = np.zeros((N))

    for c in C:

        t = [list(Nodes[c[0]]), list(Nodes[c[1]]), list(Nodes[c[2]]), list(Nodes[c[3]])]

        g_phi = grad_phi(t[0], t[1], t[2], t[3])

        for i in range(0, 3):
            # R[c[i]] = R[c[i]] 
            for j in range(0, 3):

                f = lambda x, y, z : np.matmul(np.matmul(D(x[0]),(g_phi[i])),(g_phi[j]))

                A[c[i],c[j]] = A[c[i],c[j]] + volume_integral(f, t)

    return A
    

net = net_init("ventrical1.msh")
solver(net)
print(len(net[1]))
