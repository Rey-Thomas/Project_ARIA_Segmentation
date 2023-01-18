import numpy as np
from stl import mesh
from mpl_toolkits import mplot3d
from matplotlib import pyplot

path_stl='D:/ARIA/transfer_3828543_files_d560ddcb/228_data/228_STL.stl'


# Create a new plot
figure = pyplot.figure()
axes = mplot3d.Axes3D(figure)

# Load the STL files and add the vectors to the plot
your_mesh = mesh.Mesh.from_file(path_stl)


#axes.add_collection3d(mplot3d.art3d.Poly3DCollection(your_mesh.vectors[0:368240//4]))
axes.add_collection3d(mplot3d.art3d.Poly3DCollection(your_mesh.data[0][1]))


# Auto scale to the mesh size
scale = your_mesh.points.flatten()
axes.auto_scale_xyz(scale, scale, scale)

# Show the plot to the screen
#pyplot.show()

print(np.shape(your_mesh.points))
print(np.shape(your_mesh.vectors))
print(len(your_mesh.normals))
print(len(your_mesh.v0))

# find the max dimensions, so we can know the bounding box, getting the height,
# width, length (because these are the step size)...
def find_mins_maxs(obj):
    minx = obj.x.min()
    maxx = obj.x.max()
    miny = obj.y.min()
    maxy = obj.y.max()
    minz = obj.z.min()
    maxz = obj.z.max()
    return minx, maxx, miny, maxy, minz, maxz

print(find_mins_maxs(your_mesh))

print(np.shape(your_mesh.x))


print(np.shape(your_mesh.data))
print(f'data: {your_mesh.data[0]}')
print(f'vectors: {your_mesh.vectors[0]}')
print(f'points: {your_mesh.points[0]}')
print(f'x coord: {your_mesh.x[0]}')
print(f'y coord: {your_mesh.y[0]}')
print(f'z coord: {your_mesh.z[0]}')
print(f'V0 coord: {your_mesh.v0[0]}')
print(f'V1 coord: {your_mesh.v1[0]}')
print(f'V2 coord: {your_mesh.v2[0]}')
print(f'Normal: {your_mesh.normals[0]}')
print(f'attr: {your_mesh.attr[0]}')

print(np.shape(your_mesh.data[0:2555][1]))


#pyplot.show()



