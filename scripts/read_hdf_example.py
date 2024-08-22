import h5py
import argparse
import numpy as np

if __name__ == '__main__':
	parser = argparse.ArgumentParser(description='Small example on how to work with the hdf5 outputs in python.')
	parser.add_argument('file', nargs=1, help='hdf file to open', default='.')
	args = parser.parse_args()


	f = h5py.File(args.file[0])
	root_group = f['VTKHDF']
	print("VTK Attributes: ")
	for (name, value) in root_group.attrs.items():
		print("{} : {}".format(name,value))
				
	print("\nScalars: ")
	for (name, value) in f.attrs.items():
		print("{} : {}".format(name,value))

	print("\nFields:")
	fields_group = root_group['PointData']
	for (name, value) in fields_group.items():
		print("{} : {}".format(name,value))
	
	# get specific field as numpy array
	vel = fields_group['Velocity'][()] # full path is f['VTKHDF']['PointData']['Velocity']
	print('\nVelocity norm: {}'.format(np.linalg.norm(vel)))
