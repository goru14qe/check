import h5py
import argparse
import numpy as np
import os

if __name__ == '__main__':
	parser = argparse.ArgumentParser(description='Script to extract a field from vtk-hdf files.')
	parser.add_argument('file', nargs='+', help='hdf file to open', default='.')
	parser.add_argument('--slice', help='indices of the slice to extract in standard numpy slicing syntax', default=":", type=str)
	parser.add_argument('--field', help='name of the field to extract', default='Velocity')
	parser.add_argument('--view', action="store_true", help='just list contents of the file')
	args = parser.parse_args()

#	for f in glob.glob(args.path[0]):
#		print(f)
#		os.path.file
#	exit()
	for filename in args.file:
		print(f"Processing file {filename}")

		f = h5py.File(filename)
		root_group = f['VTKHDF']
		fields_group = root_group['PointData']

		if args.view:
			print("VTK Attributes: ")
			for (name, value) in root_group.attrs.items():
				print("{} : {}".format(name,value))
						
			print("\nScalars: ")
			for (name, value) in f.attrs.items():
				print("{} : {}".format(name,value))

			print("\nFields:")
			for (name, value) in fields_group.items():
				print("{} : {}".format(name,value))
			exit()

		#extend = root_group.attrs["WholeExtent"]
		spacing = root_group.attrs["Spacing"]
		origin = root_group.attrs["Origin"]
		
		# get specific field as numpy array
		valid_chars = "1234567890-,: "
		for c in args.slice:
			if not c in valid_chars:
				print(f'Encountered invalid char "{c}" in slice. Must be one of "{valid_chars}".')
				exit()
		field = fields_group[args.field][()]
		if len(field.shape) == 3:
			field = np.expand_dims(field, axis=3)

		# compute coordinates for all nodes
		posx = np.linspace(origin[0], origin[0] + spacing[0] * field.shape[0], field.shape[0])
		posy = np.linspace(origin[1], origin[1] + spacing[1] * field.shape[1], field.shape[1])
		posz = np.linspace(origin[2], origin[2] + spacing[2] * field.shape[2], field.shape[2])
		x,y,z = np.meshgrid(posx,posy,posz, indexing='ij')

		# select slice
		field = eval(f'field[{args.slice}]')
		x = eval(f'x[{args.slice}]')
		y = eval(f'y[{args.slice}]')
		z = eval(f'z[{args.slice}]')

		# write flattened list of nodes to csv
		coords = np.stack((x,y,z), axis=-1)
		data = np.concatenate((coords, field), axis=-1)
		data = data.reshape(-1, data.shape[-1])
		name = os.path.splitext(filename)[0]
		name = f'{name}.csv'
		print(f"Writing output {name}")
		np.savetxt(name, data, delimiter=',')

	#print('\nVelocity norm: {}'.format(np.linalg.norm(vel)))
