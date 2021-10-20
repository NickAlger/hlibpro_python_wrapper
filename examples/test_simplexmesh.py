import numpy as np
import hlibpro_python_wrapper as hpro
from time import time
import matplotlib.pyplot as plt
from scipy.spatial import KDTree
import dolfin as dl

from nalger_helper_functions import circle_mesh

hcpp = hpro.hpro_cpp

npts = 3
dim = 2

points = np.random.randn(dim,npts)
query = np.random.randn(dim)

coords = hcpp.projected_affine_coordinates(query, points)

print('coords=', coords)

err_coords = np.linalg.norm(query - np.dot(points, coords))
print('err_coords=', err_coords)

err_affine_constraint = np.abs(1. - np.sum(coords))
print('err_affine_constraint=', err_affine_constraint)


# CLOSEST POINT TO LINE SEGMENT

npts = 2
dim = 2
nquery = 100

segment_vertices = np.random.randn(dim, npts)

plt.figure()
plt.plot(segment_vertices[0,:], segment_vertices[1,:])

qq = np.random.randn(dim, nquery)
SS = segment_vertices.T.reshape((-1,1)) * np.ones(nquery)
pp = hcpp.closest_point_in_simplex_vectorized(qq, SS)

for k in range(30):
    query = qq[:,k]
    closest_point = pp[:,k]
    plt.plot([query[0], closest_point[0]], [query[1], closest_point[1]], 'gray')
    plt.plot(query[0], query[1], '*r')
    plt.plot(closest_point[0], closest_point[1], '.k')

plt.gca().set_aspect('equal')


# CLOSEST POINT TO TRIANGLE

npts = 3
dim = 2
nquery = 100

triangle_vertices = np.random.randn(dim, npts)

plt.figure()
plt.plot([triangle_vertices[0,0], triangle_vertices[0,1], triangle_vertices[0,2], triangle_vertices[0,0]],
         [triangle_vertices[1,0], triangle_vertices[1,1], triangle_vertices[1,2], triangle_vertices[1,0]])

qq = np.random.randn(dim, nquery)
SS = triangle_vertices.T.reshape((-1,1)) * np.ones(nquery)
pp = hcpp.closest_point_in_simplex_vectorized(qq, SS)

for k in range(nquery):
    query = qq[:,k]
    closest_point = pp[:,k]
    plt.plot([query[0], closest_point[0]], [query[1], closest_point[1]], 'gray')
    plt.plot(query[0], query[1], '*r')
    plt.plot(closest_point[0], closest_point[1], '.k')

plt.gca().set_aspect('equal')


# TIMING

npts = 3
dim = 2
nquery = int(1e6)

qq = np.random.randn(dim, nquery)
SS = np.random.randn(dim*npts, nquery)

t = time()
pp = hcpp.closest_point_in_simplex_vectorized(qq, SS)
dt_closest_point = time() - t
print('nquery=', nquery, ', dt_closest_point=', dt_closest_point)


# POINT IN MESH TEST

nquery = 5000

mesh = circle_mesh(np.array([0.0, 0.0]), 1.0, 0.25)
SM = hcpp.SimplexMesh2D(mesh.coordinates(), mesh.cells())

plt.figure()
dl.plot(mesh)

qq = np.random.randn(nquery, 2)
in_mesh = SM.point_is_in_mesh_vectorized(qq)
not_in_mesh = np.logical_not(in_mesh)

plt.plot(qq[in_mesh,0], qq[in_mesh,1], '.r')
plt.plot(qq[not_in_mesh,0], qq[not_in_mesh,1], '.k')


# CLOSEST POINT TO MESH

nquery = 100

mesh = circle_mesh(np.array([0.0, 0.0]), 1.0, 0.25)
SM = hcpp.SimplexMesh2D(mesh.coordinates(), mesh.cells())

plt.figure()
dl.plot(mesh)

qq = np.random.randn(nquery, 2)
pp = SM.closest_point_vectorized(qq);

for k in range(nquery):
    query = qq[k,:]
    closest_point = pp[k,:]
    plt.plot([query[0], closest_point[0]], [query[1], closest_point[1]], 'gray')
    plt.plot(query[0], query[1], '*r')
    plt.plot(closest_point[0], closest_point[1], '.k')

plt.gca().set_aspect('equal')


# CLOSEST POINT TO MESH TIMING

nquery = int(1e5)
mesh_h = 1e-2

mesh = circle_mesh(np.array([0.0, 0.0]), 1.0, mesh_h)
num_cells = mesh.cells().shape[0]

SM = hcpp.SimplexMesh2D(mesh.coordinates(), mesh.cells())

t = time()
dt_build_SM = time() - t
print('num_cells=', num_cells, ', dt_build_SM=', dt_build_SM)

qq = 0.7*np.random.randn(nquery, 2)
fraction_outside_mesh = np.sum(np.linalg.norm(qq, axis=1) >= 1) / nquery
print('fraction_outside_mesh=', fraction_outside_mesh)

t = time()
pp = SM.closest_point_vectorized(qq);
dt_query_SM = time() - t
print('nquery=', nquery, ', dt_query_SM=', dt_query_SM)

# Without inside mesh pre-check:
# num_cells= 39478 , dt_build_SM= 7.367134094238281e-05
# nquery= 100000 , dt_query_SM= 2.591938018798828

# With inside mesh pre-check:
# fraction_outside_mesh= 0.35976
# num_cells= 39478 , dt_build_SM= 3.170967102050781e-05
# nquery= 100000 , dt_query_SM= 0.9992811679840088


# EVALUATE FUNCTION AT POINT

mesh = circle_mesh(np.array([0.0, 0.0]), 1.0, 1e-2)
V = dl.FunctionSpace(mesh, 'CG', 1)
mesh_coords = mesh.coordinates()
dof_coords = V.tabulate_dof_coordinates()
dof2vertex = dl.dof_to_vertex_map(V)
vertex2dof = dl.vertex_to_dof_map(V)

u = dl.Function(V)
u.vector()[:] = dof_coords[:,0]**2 + 2*dof_coords[:,1]**2
u.set_allow_extrapolation(True)

uu = u.vector()[vertex2dof]
# uu = u.vector()[dof2vertex]

uu_true = np.zeros(V.dim())
for ii in range(V.dim()):
    uu_true[ii] = u(mesh_coords[ii,:])

err_uu = np.linalg.norm(uu - uu_true)
print('err_uu=', err_uu)

p = np.random.randn(2)
p = p / 4.;
u_of_p_true = u(p)

SM = hcpp.SimplexMesh2D(mesh.coordinates(), mesh.cells())
u_of_p = SM.evaluate_function_at_point(uu, p)
print('p=', p, ', u_of_p=', u_of_p, ', u_of_p_true=', u_of_p_true)

#

nquery = int(1e5)
pp = np.random.rand(nquery, 2)

t = time()
upp = SM.evaluate_function_at_point_vectorized(uu, pp)
dt_eval = time() - t
# print('V.dim()=', V.dim(), ', nquery=', nquery, ', dt_eval=', dt_eval)

t = time()
upp_true = np.zeros(nquery)
for ii in range(nquery):
    p = pp[ii,:]
    if mesh.bounding_box_tree().compute_first_entity_collision(dl.Point(p)) < mesh.cells().shape[0]:
        upp_true[ii] = u(p)
dt_eval_fenics = time() - t
print('V.dim()=', V.dim(), ', nquery=', nquery, ', dt_eval=', dt_eval, 'dt_eval_fenics=', dt_eval_fenics)

err_eval_function = np.linalg.norm(upp - upp_true)
print('err_eval_function=', err_eval_function)