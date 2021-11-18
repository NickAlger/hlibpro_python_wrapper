import numpy as np
import hlibpro_python_wrapper as hpro
from time import time, sleep
import matplotlib.pyplot as plt
from scipy.spatial import KDTree
import dolfin as dl

from nalger_helper_functions import circle_mesh, plot_ellipse

hcpp = hpro.hpro_cpp

# npts = 3
# dim = 2
#
# points = np.random.randn(dim,npts)
# query = np.random.randn(dim)
#
# coords = hcpp.projected_affine_coordinates(query, points)
#
# print('coords=', coords)
#
# err_coords = np.linalg.norm(query - np.dot(points, coords))
# print('err_coords=', err_coords)
#
# err_affine_constraint = np.abs(1. - np.sum(coords))
# print('err_affine_constraint=', err_affine_constraint)


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
nquery = int(1e5)

qq = np.random.randn(dim, nquery)
SS = np.random.randn(dim*npts, nquery)

t = time()
pp = hcpp.closest_point_in_simplex_vectorized(qq, SS)
dt_closest_point = time() - t
print('nquery=', nquery, ', dt_closest_point=', dt_closest_point)


# POINT IN MESH TEST

nquery = 5000

mesh = circle_mesh(np.array([0.0, 0.0]), 1.0, 0.25)
vertices = np.array(mesh.coordinates().T, order='F')
cells = np.array(mesh.cells().T, order='F')

SM = hcpp.SimplexMesh(vertices, cells)

plt.figure()
dl.plot(mesh)

qq = np.array(np.random.randn(2, nquery), order='F')
in_mesh = SM.point_is_in_mesh_vectorized(qq)
not_in_mesh = np.logical_not(in_mesh)

plt.plot(qq[0,in_mesh], qq[1,in_mesh], '.r')
plt.plot(qq[0,not_in_mesh], qq[1,not_in_mesh], '.k')

print('asdf1')

# CLOSEST POINT TO MESH

nquery = 100

mesh = circle_mesh(np.array([0.0, 0.0]), 1.0, 0.25)
vertices = np.array(mesh.coordinates().T, order='F')
cells = np.array(mesh.cells().T, order='F')
SM = hcpp.SimplexMesh(vertices, cells)

plt.figure()
dl.plot(mesh)

qq = np.array(np.random.randn(2, nquery), order='F')
pp = SM.closest_point_vectorized(qq)

for k in range(nquery):
    query = qq[:,k]
    closest_point = pp[:,k]
    plt.plot([query[0], closest_point[0]], [query[1], closest_point[1]], 'gray')
    plt.plot(query[0], query[1], '*r')
    plt.plot(closest_point[0], closest_point[1], '.k')

plt.gca().set_aspect('equal')


# CLOSEST POINT TO MESH TIMING

nquery = int(1e5)
mesh_h = 1e-2

mesh = circle_mesh(np.array([0.0, 0.0]), 1.0, mesh_h)
vertices = np.array(mesh.coordinates().T, order='F')
cells = np.array(mesh.cells().T, order='F')
num_cells = cells.shape[1]

t = time()
SM = hcpp.SimplexMesh(vertices, cells)
dt_build_SM = time() - t
print('num_cells=', num_cells, ', dt_build_SM=', dt_build_SM)

qq = 0.7*np.array(np.random.randn(2, nquery), order='F')
fraction_outside_mesh = np.sum(np.linalg.norm(qq, axis=0) >= 1) / nquery
print('fraction_outside_mesh=', fraction_outside_mesh)

SM.set_sleep_duration(10)
SM.set_thread_count(8)
sleep(1) # Wait for change to take effect

t = time()
pp = SM.closest_point_vectorized(qq)
dt_closest_SM = time() - t
print('nquery=', nquery, ', dt_closest_SM=', dt_closest_SM)

SM.reset_thread_count_to_default()
SM.reset_sleep_duration_to_default()

# t = time()
# pp_multi = SM.closest_point_vectorized_multithreaded(qq)
# dt_closest_SM_multi = time() - t
# print('nquery=', nquery, ', dt_closest_SM_multi=', dt_closest_SM_multi)
#
# err_multi = np.linalg.norm(pp_multi - pp)
# print('err_multi=', err_multi)

# Without inside mesh pre-check:
# num_cells= 39478 , dt_build_SM= 7.367134094238281e-05
# nquery= 100000 , dt_query_SM= 2.591938018798828

# With inside mesh pre-check:
# fraction_outside_mesh= 0.35976
# num_cells= 39478 , dt_build_SM= 3.170967102050781e-05
# nquery= 100000 , dt_query_SM= 0.9992811679840088

# With precomputed facet stuff
# nquery= 100000 , dt_query_SM= 0.6365664005279541

# With boundary search
# nquery= 100000 , dt_closest_SM= 0.1751554012298584

# With multithreading
# nquery= 100000 , dt_closest_SM_multi= 0.0556492805480957s

# No multithreading, no geometric ordering
# closest_point_vectorized() took 220725 microseconds
# nquery= 100000 , dt_closest_SM= 0.1983785629272461
# and
# closest_point_vectorized() took 1940110 microseconds
# nquery= 1000000 , dt_closest_SM= 1.940319299697876

# no multithreading, geometric ordering
# closest_point_vectorized() took 157871 microseconds
# nquery= 100000 , dt_closest_SM= 0.2701895236968994
# and
# closest_point_vectorized() took 1249172 microseconds
# nquery= 1000000 , dt_closest_SM= 3.3071250915527344

# EVALUATE FUNCTION AT POINT

mesh = circle_mesh(np.array([0.0, 0.0]), 1.0, 1e-2)
V = dl.FunctionSpace(mesh, 'CG', 1)

# check that I'm using the dof to vertex mapping right

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

#

vertices = np.array(mesh.coordinates().T, order='F')
cells = np.array(mesh.cells().T, order='F')
SM = hcpp.SimplexMesh(vertices, cells)

nquery = int(1e5)
num_functions = 20
pp = np.array(np.random.rand(2,nquery), order='F')

uu = [dl.Function(V) for _ in range(num_functions)]
for u in uu:
    u.vector()[:] = dof_coords[:,0]**2 + 2*dof_coords[:,1]**2
    u.set_allow_extrapolation(True)

UU = np.array([u.vector()[vertex2dof] for u in uu], order='F')

t = time()
upp = SM.evaluate_CG1_functions_at_points(UU, pp)
dt_eval = time() - t
print('V.dim()=', V.dim(), ', nquery=', nquery, ', num_functions=', num_functions, ', dt_eval=', dt_eval)

t = time()
upp_true = np.zeros((num_functions, nquery))
for ii in range(nquery):
    p = pp[:,ii]
    if mesh.bounding_box_tree().compute_first_entity_collision(dl.Point(p)) < mesh.cells().shape[0]:
        for jj in range(num_functions):
            upp_true[jj, ii] = uu[jj](p)
dt_eval_fenics = time() - t
print('V.dim()=', V.dim(), ', nquery=', nquery, ', num_functions=', num_functions, ', dt_eval=', dt_eval, 'dt_eval_fenics=', dt_eval_fenics)

err_eval_function = np.linalg.norm(upp - upp_true) / np.linalg.norm(upp_true)
print('err_eval_function=', err_eval_function)

# No multithreading:
# V.dim()= 20054 , nquery= 100000 , num_functions= 20 , dt_eval= 0.10272383689880371 dt_eval_fenics= 23.443379878997803

# With multithreading:
# V.dim()= 20054 , nquery= 100000 , num_functions= 20 , dt_eval= 0.025378704071044922 dt_eval_fenics= 24.46290874481201


# EVALUATE FUNCTION AT POINT USING REFLECTION

mesh = circle_mesh(np.array([0.0, 0.0]), 1.0, 1e-2)
V = dl.FunctionSpace(mesh, 'CG', 1)

mesh_coords = mesh.coordinates()
dof_coords = V.tabulate_dof_coordinates()
dof2vertex = dl.dof_to_vertex_map(V)
vertex2dof = dl.vertex_to_dof_map(V)

vertices = np.array(mesh.coordinates().T, order='F')
cells = np.array(mesh.cells().T, order='F')
SM = hcpp.SimplexMesh(vertices, cells)

nx = 100
ny = 100
xmin = -2.0
xmax = 2.0
ymin = -2.0
ymax = 2.0
X, Y = np.meshgrid(np.linspace(xmin, xmax, nx), np.linspace(ymin, ymax, ny))
pp = np.array([X.reshape(-1), Y.reshape(-1)], order='F')
nquery = pp.shape[1]

uu = list()
uu.append(dl.interpolate(dl.Expression('sin(x[0])', degree=3), V))
uu.append(dl.interpolate(dl.Expression('cos(2*x[1] + x[0])', degree=3), V))
uu.append(dl.interpolate(dl.Expression('sin(pow(x[0],2) + pow(x[1],2))', degree=3), V))
uu.append(dl.interpolate(dl.Expression('sin(pow((x[0]-0.2)*2,2) + pow((x[1]-0.5)*3,2))', degree=3), V))
uu.append(dl.interpolate(dl.Expression('sin(8*x[0]) * cos(9*x[1])', degree=3), V))

num_functions = len(uu)

UU = np.array([u.vector()[vertex2dof] for u in uu], order='F')

t = time()
Z = SM.evaluate_functions_at_points_with_reflection(UU, pp).reshape((num_functions, nx, ny))
dt_eval_reflection = time() - t
print('V.dim()=', V.dim(), ', nquery=', nquery, ', num_functions=', num_functions, ', dt_eval_reflection=', dt_eval_reflection)

for ii in range(num_functions):
    plt.figure()
    plt.subplot(1,2,1)
    dl.plot(uu[ii])
    plt.xlim(xmin, xmax)
    plt.ylim(ymin, ymax)
    plt.gca().set_aspect('equal')
    plt.title('original')

    plt.subplot(1,2,2)
    plt.pcolor(X, Y, Z[ii, :, :], shading='auto')
    plt.xlim(xmin, xmax)
    plt.ylim(ymin, ymax)
    plt.gca().set_aspect('equal')
    plt.title('with reflection')

# simplex ind and affine coords getter inside function
# V.dim()= 20054 , nquery= 10000 , num_functions= 5 , dt_eval_reflection= 0.014193296432495117

# simplex ind and affine coords getter outside function
# V.dim()= 20054 , nquery= 10000 , num_functions= 5 , dt_eval_reflection= 0.01358175277709961

# EVALUATE FUNCTION AT POINT USING REFLECTION AND ELLIPSOID TRUNCATION

mu1 = np.array([0.26, 0.25])
Sigma1 = 0.5*np.array([[1., 0.2],[0.2, 1.]])

mu2 = np.array([-0.65, 0.1])
Sigma2 = 0.25*np.array([[0.25, -0.1],[-0.1, 1.0]])

all_mu = [mu1, mu2]
all_inv_Sigma = [np.linalg.inv(Sigma1), np.linalg.inv(Sigma2)]

tau = 1.

mesh = circle_mesh(np.array([0.0, 0.0]), 1.0, 1e-2)
V = dl.FunctionSpace(mesh, 'CG', 1)

mesh_coords = mesh.coordinates()
dof_coords = V.tabulate_dof_coordinates()
dof2vertex = dl.dof_to_vertex_map(V)
vertex2dof = dl.vertex_to_dof_map(V)

vertices = np.array(mesh.coordinates().T, order='F')
cells = np.array(mesh.cells().T, order='F')
SM = hcpp.SimplexMesh(vertices, cells)

nx = 100
ny = 100
xmin = -2.0
xmax = 2.0
ymin = -2.0
ymax = 2.0
X, Y = np.meshgrid(np.linspace(xmin, xmax, nx), np.linspace(ymin, ymax, ny))
pp = np.array([X.reshape(-1), Y.reshape(-1)], order='F')
nquery = pp.shape[1]

uu = list()
uu.append(dl.interpolate(dl.Expression('sin(pow((x[0]-0.2)*2,2) + pow((x[1]-0.5)*3,2))', degree=3), V))
uu.append(dl.interpolate(dl.Expression('sin(8*x[0]) * cos(9*x[1])', degree=3), V))

num_functions = len(uu)

UU = np.array([u.vector()[vertex2dof] for u in uu], order='F')

t = time()
Z = SM.evaluate_functions_at_points_with_reflection_and_ellipsoid_truncation(UU, pp, all_mu, all_inv_Sigma, tau).reshape((num_functions, nx, ny))
dt_eval_etrunc = time() - t
print('V.dim()=', V.dim(), ', nquery=', nquery, ', num_functions=', num_functions, ', dt_eval_etrunc=', dt_eval_etrunc)

for ii in range(num_functions):
    plt.figure()
    plt.subplot(1,2,1)
    dl.plot(uu[ii])
    plt.xlim(xmin, xmax)
    plt.ylim(ymin, ymax)
    plt.gca().set_aspect('equal')
    plt.title('original')

    plt.subplot(1,2,2)
    plt.pcolor(X, Y, Z[ii, :, :], shading='auto')
    plot_ellipse(np.zeros(2), np.eye(2), 1.0)
    plot_ellipse(all_mu[ii], np.linalg.inv(all_inv_Sigma[ii]), tau)
    plt.xlim(xmin, xmax)
    plt.ylim(ymin, ymax)
    plt.gca().set_aspect('equal')
    plt.title('with reflection')

# TIMING AFTER REMOVING TEMPLATED DIMENSION K
# nquery= 100000 , dt_closest_point= 0.8269867897033691
# asdf1
# num_cells= 39478 , dt_build_SM= 0.24170398712158203
# fraction_outside_mesh= 0.3627
# nquery= 100000 , dt_closest_SM= 0.10088753700256348
# err_uu= 1.808961479051732e-14
# V.dim()= 20054 , nquery= 100000 , num_functions= 20 , dt_eval= 0.048201560974121094
# V.dim()= 20054 , nquery= 100000 , num_functions= 20 , dt_eval= 0.048201560974121094 dt_eval_fenics= 24.346689701080322
# err_eval_function= 9.178371715472322e-15
# V.dim()= 20054 , nquery= 10000 , num_functions= 5 , dt_eval_reflection= 0.04037642478942871
# V.dim()= 20054 , nquery= 10000 , num_functions= 2 , dt_eval_etrunc= 0.03749442100524902


# TIMING BEFORE REMOVING TEMPLATED DIMENSION K
# coords= [-2.26117588  3.6123003  -0.35112442]
# err_coords= 2.220446049250313e-16
# err_affine_constraint= 2.220446049250313e-16
# nquery= 100000 , dt_closest_point= 0.8168816566467285
# asdf1
# num_cells= 39478 , dt_build_SM= 0.24517488479614258
# fraction_outside_mesh= 0.35912
# nquery= 100000 , dt_closest_SM= 0.07282233238220215
# err_uu= 1.808961479051732e-14
# V.dim()= 20054 , nquery= 100000 , num_functions= 20 , dt_eval= 0.04496169090270996
# V.dim()= 20054 , nquery= 100000 , num_functions= 20 , dt_eval= 0.04496169090270996 dt_eval_fenics= 23.492988348007202
# err_eval_function= 9.253898163209867e-15
# V.dim()= 20054 , nquery= 10000 , num_functions= 5 , dt_eval_reflection= 0.028449296951293945
# V.dim()= 20054 , nquery= 10000 , num_functions= 2 , dt_eval_etrunc= 0.0235598087310791