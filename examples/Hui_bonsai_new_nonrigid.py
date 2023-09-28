
use_cuda = True
if use_cuda:
    import cupy as cp
    to_cpu = cp.asnumpy
    cp.cuda.set_allocator(cp.cuda.MemoryPool().malloc)
else:
    cp = np
    to_cpu = lambda x: x

import time

from plyfile import PlyData
import numpy as np
import open3d as o3
import transforms3d as t3d
from probreg import callbacks, cpd, bcpd
import matplotlib.pyplot as plt

import utils
# import logging
# log = logging.getLogger('probreg')
# log.setLevel(logging.DEBUG)


path = r'C:\Users\WANGH0M\gaussian-splatting\output'

old_path = path + r'\bonsai_old_cut.ply'
new_path = path + r'\bonsai_new_cut.ply'

old_filter_path = path + r'\bonsai_old_cut_filtered.ply' ##source
new_filter_path = path + r'\bonsai_new_cut_filtered.ply' ##target

def vertex_normals(filter_path):
    filter = PlyData.read(filter_path)
    normals = np.array([list(point) for point in filter['vertex'].data])[:,3:6] ##choose only first 3 columns, otherwise it has 62 columns
    return normals

old_normals = vertex_normals(old_filter_path)
new_normals = vertex_normals(new_filter_path)
#print(old_normals.shape, new_normals.shape) ##same shape as the vertices

cropped_pcd = o3.io.read_point_cloud(old_path)
old_points = np.asarray(cropped_pcd.points)

cropped_pcd = o3.io.read_point_cloud(new_path)
new_points = np.asarray(cropped_pcd.points)
print(old_points.shape, new_points.shape)

def prepare_source_and_target_nonrigid_3d__Hui(source_pts, target_pts, voxel_size=5.0):
    source = o3.geometry.PointCloud()
    target = o3.geometry.PointCloud()
    source.points = o3.utility.Vector3dVector(source_pts)
    target.points = o3.utility.Vector3dVector(target_pts)
    source = source.voxel_down_sample(voxel_size=voxel_size)
    target = target.voxel_down_sample(voxel_size=voxel_size)
    return source, target


if 1:
    source, target = prepare_source_and_target_nonrigid_3d__Hui(old_points, new_points, 0.1)
elif 0: ##NO USE! only one input as source, can be .txt or .ply; 
    source, target = utils.prepare_source_and_target_rigid_3d(new_path, 0.1)

source_pt = cp.asarray(source.points, dtype=cp.float32)
target_pt = cp.asarray(target.points, dtype=cp.float32)

start = time.time()

cbs = [callbacks.Open3dVisualizerCallback(source, target)]
if 1:
    from probreg import filterreg
    tf_param, _, _ = filterreg.registration_filterreg(source, target, new_normals,
                                                    objective_type="pt2pt" ,
                                                    sigma2=None, update_sigma2=True,
                                                    maxiter=100, tol=0.0001,
                                                    callbacks=cbs)
    "https://probreg.readthedocs.io/en/latest/_modules/probreg/transformation.html#RigidTransformation.inverse"
    print(tf_param) ##transformation: rotation rot, scale, translation t
    print(tf_param.inverse) ##RigidTransformation(self.rot.T, -self.xp.dot(self.rot.T, self.t) / self.scale, 1.0 / self.scale)
    print("result: ", np.rad2deg(t3d.euler.mat2euler(tf_param.rot)), tf_param.scale, tf_param.t)

elif 0:
    tf_param, _, _ = cpd.registration_cpd(source, target, callbacks=cbs)
    print("result: ", np.rad2deg(t3d.euler.mat2euler(tf_param.rot)), tf_param.scale, tf_param.t)
    
elif 0:
    tf_param = bcpd.registration_bcpd(source, target, callbacks=cbs)
    print("result: ", np.rad2deg(t3d.euler.mat2euler(tf_param.rot)), tf_param.scale, tf_param.t)

elif 0:
    acpd = cpd.AffineCPD(source_pt, use_cuda=use_cuda)
    tf_param, _, _ = acpd.registration(target_pt)
    print("result: ", to_cpu(tf_param.b), to_cpu(tf_param.t))

elif 0:
    acpd = cpd.NonRigidCPD(source_pt, use_cuda=use_cuda)
    tf_param, _, _ = acpd.registration(target_pt)

    # result = tf_param.transform(source_pt)
    # pc = o3.geometry.PointCloud()
    # pc.points = o3.utility.Vector3dVector(to_cpu(result))
    # pc.paint_uniform_color([0, 1, 0])
    # target.paint_uniform_color([0, 0, 1])
    # o3.visualization.draw_geometries([pc, target])

    print("result: ", to_cpu(tf_param.w), to_cpu(tf_param.g))


plt.show()

elapsed = time.time() - start
print("time: ", elapsed)