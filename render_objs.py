import trimesh
import DracoPy
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import argparse

def open_obj(object_path):
    return trimesh.load_mesh(object_path)
    # # Get the TextureVisuals object
    # texture_visuals = mesh.visual.to_texture()
    # # Access the UV coordinates from the TextureVisuals object
    # uv_coords = texture_visuals.uv
    # texture_image = texture_visuals.image

def get_texture_at_uv(uv, texture_image):
    # Map the UV coordinate to the pixel coordinate
    # UV coordinates are in [0, 1], so we multiply by image dimensions to get pixel coordinates
    u, v = uv
    x = int(u * texture_image.shape[1])
    y = int(v * texture_image.shape[0])

    # Note: image arrays are typically accessed as (y, x) not (x, y)
    # Sample the texture image at the pixel coordinate to get the color
    color = texture_image[y, x]

    return color

# uv_coord = [0.6, 0.7]
# color = get_texture_at_uv(uv_coord, texture_image)
# print(f"Color at UV coordinate {uv_coord} is {color}")


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--file_path", "-f", help="file where the data is present", required=True)
    args = parser.parse_args()
    return args

# mesh = open_drc("geo/rom_joints/drc/rom_joints.00001.drc")
args = parse_arguments()
mesh = open_obj(args.file_path)
# You can now export in other formats or use as is
# mesh.export('output.obj')  # or output.ply, output.stl, etc.
# mesh.export(f"render_{args.file_path.split('.')[-3].split('/')[-1]}_{args.file_path.split('.')[-2]}.ply", file_type="ply")

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