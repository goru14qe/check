import numpy as np
from prettytable import PrettyTable
import argparse

parser = argparse.ArgumentParser(description='Generate possible domain decompositions.')
parser.add_argument('--num_proc',type=int, help='number of target processes')
parser.add_argument('--domain', nargs='+', default=[64,64,64], help='size of the domain in nodes')
parser.add_argument('--filter', action=argparse.BooleanOptionalAction, default=True, help='only display promising decompositions')
parser.add_argument('--num_proc_min', default=0, type=int, help='minimum number of processor nodes to use, if a value > 0 is given then all possibilities from num_proc_min to num_proc are considered')
args = parser.parse_args()

# parameters
domain = np.array(args.domain).astype(int)
filter_results = args.filter
dims = len(domain)

if args.num_proc_min > 0:
	num_processes = range(args.num_proc, args.num_proc_min, -1)
else:
	num_processes = [int(args.num_proc)]

def cartesian_product(*arrays):
	ndim = len(arrays)
	return (np.stack(np.meshgrid(*arrays), axis=-1).reshape(-1, ndim))

chunk_size_all = []
total_surface_all = []
decompositions_all = []
min_surface_per_num_processes = {}
for num_proc in num_processes:
	# all integer divisors
	divisors = np.array(list(filter(lambda i: num_proc % i == 0, range(1, num_proc+1))))
	divisors = [divisors] * dims
	# [divisors] + [divisors[0:-2]] * (dims-1)

	decompositions = cartesian_product(*divisors)
	# for square domains the order does not matter
	if filter_results and np.all(domain == domain[0]):
		decompositions = np.sort(decompositions, axis=1)[:,::-1]
		decompositions = np.unique(decompositions, axis=0)
	inds = np.argsort(decompositions[:,-1], axis=0)
	decompositions = decompositions[inds]

	num_chunks = np.prod(decompositions, axis=1)
	decompositions = decompositions[num_chunks == num_proc]

	chunk_size = domain / decompositions
	total_surface = (np.sum(chunk_size * 2, axis=1) * num_proc).astype(int)

	inds = np.argsort(total_surface, kind='stable')
	chunk_size_all.append(chunk_size[inds])
	total_surface_all.append(total_surface[inds])
	decompositions_all.append(decompositions[inds])
	min_surface_per_num_processes[num_proc] = total_surface[inds[0]]

chunk_size = np.concatenate(chunk_size_all)
total_surface = np.concatenate(total_surface_all)
decompositions = np.concatenate(decompositions_all)
num_proc_total = np.prod(decompositions, 1)

min_surface = np.min(total_surface)
min_proc_total = np.min(num_proc_total)
min_chunk_z = np.min(chunk_size[:,2])

#inds = np.argsort(total_surface, kind='stable')
if filter_results:
	inds_filtered = []
	for i in range(0, len(total_surface)):
		# skip entries which are just permutations of the previous one
		if (len(inds_filtered) > 0 
			and total_surface[i] == total_surface[inds_filtered[-1]]
			and num_proc_total[i] == num_proc_total[inds_filtered[-1]]):
			continue
		# look for pareto front
		better_or_equal = np.logical_and(np.logical_and(num_proc_total >= num_proc_total[i], total_surface <= total_surface[i]), 
		 chunk_size[:,2] >=  chunk_size[i,2])
		strictly_better = np.logical_or(np.logical_or(num_proc_total > num_proc_total[i], total_surface < total_surface[i]),
		 chunk_size[:,2] >  chunk_size[i,2])
		is_dominated = np.any(np.logical_and(better_or_equal, strictly_better))
		if (not is_dominated and (total_surface[i] == min_surface_per_num_processes[num_proc_total[i]]
				or (total_surface[i] < 3 * min_surface and chunk_size[i,2] > 150))): # general cutoff for to much synchronization and threshold where vectorization speedup can be expected
			inds_filtered.append(i)

	# always put in smallest surface decomposition
	# inds_filtered = [min_surface_idx]
	# for i in range(0, len(total_surface)):
	# 	idx = i
	# 	if idx == min_surface_idx:
	# 		continue
	# 	# heuristic for promising candidates
	# 	if (total_surface[idx] < total_surface[inds_filtered[-1]] # indicates fewer processors
	# 		or (total_surface[idx] != total_surface[inds_filtered[-1]]
	# 		and np.all(chunk_size[idx,2] > chunk_size[inds_filtered,2])
    #   		and total_surface[idx] < 3 * min_surface # general cutoff for to much synchronization
	# 		and chunk_size[idx,2] > 150)): # threshold where vectorization speedup can be expected
	# 		inds_filtered.append(idx)
	inds = np.array(inds_filtered)

decompositions = decompositions[inds]
total_surface = total_surface[inds]
chunk_size = chunk_size[inds].astype(int)
num_proc_total = num_proc_total[inds]

#print("nodes per core: {}".format(int(np.prod(domain) / num_processes)))
table = PrettyTable()
dim_names = ['X', 'Y', 'Z']
#table.field_names = dim_names[0:dims] + ['surface nodes']
table.align = 'l'
for i in range(0, dims):
	table.add_column(dim_names[i], decompositions[:,i])
if len(num_processes) > 1:
	table.add_column('n', num_proc_total)
	table.add_column('nodes per core', (np.prod(domain) / num_proc_total).astype(int))
table.add_column('surface nodes', total_surface)
#table.add_column('nodes X', chunk_size[:,0])
#table.add_column('nodes Y', chunk_size[:,1])
table.add_column('nodes Z', chunk_size[:,2])

print(table)
