# requirements: argparse, h5py
import argparse
import h5py
import os

# this script is only useful for hdf files created before 
# commit a19d6635c870a0aa081983d98e7a3b7188e6cb76 'workaround for vtkhdf string attrs'
parser = argparse.ArgumentParser(description='Fix the vtk type of hdf5 files so that paraview can open them.')
parser.add_argument('directory', nargs=1, help='directory which will be recursively searched for .hdf files', default='.')
args = parser.parse_args()

directory = args.directory[0]
for root, dirs, files in os.walk(directory):
	for filename in files:
		if filename.endswith(".hdf"):
			full_name = os.path.join(root, filename)
			print("Fixing {}".format(filename))
			file = h5py.File(full_name, "r+")
			type_attr = file["VTKHDF"].attrs.create("Type", "ImageData", dtype=h5py.string_dtype(encoding='ascii', length=9))
