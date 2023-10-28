import trimesh
import DracoPy
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import argparse

# We use drc as they are smaller than obj. This function imports them as trimesh objects
def open_drc(object_path):
    with open(object_path, 'rb') as draco_file:
        mesh_object = DracoPy.decode_buffer_to_mesh(draco_file.read())
        
    vertices = np.array(mesh_object.points).astype(np.float32)
    if hasattr(mesh_object, 'faces'):
        faces = np.array(mesh_object.faces).astype(np.uint32)
    else:
        faces = None
    if hasattr(mesh_object, 'tex_coord'):
        visual = trimesh.visual.TextureVisuals( uv = np.array(mesh_object.tex_coord).astype(np.float32) )
    else:
        visual = None
    
    return trimesh.Trimesh(vertices=vertices, faces=faces, visual=visual )


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--file_path", "-f", help="file where the data is present", required=True)
    args = parser.parse_args()
    return args

# mesh = open_drc("geo/rom_joints/drc/rom_joints.00001.drc")
args = parse_arguments()
mesh = open_drc(args.file_path)
# You can now export in other formats or use as is
# mesh.export('output.obj')  # or output.ply, output.stl, etc.
mesh.export(f"render_{args.file_path.split('.')[-3].split('/')[-1]}_{args.file_path.split('.')[-2]}.ply", file_type="ply")

# Lets just preview the mesh inefficently in matplotlib
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# if your mesh has faces
if mesh.faces is not None:
    poly3d = [mesh.vertices[face] for face in mesh.faces]
    ax.add_collection3d(Poly3DCollection(poly3d, alpha=.25, linewidths=0.1, edgecolors='r', facecolor='cyan'))

ax.scatter(mesh.vertices[:, 0], mesh.vertices[:, 1], mesh.vertices[:, 2])  # if you just want to plot vertices
# plt.show()
plt.savefig('render_mesh.jpg')