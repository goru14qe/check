# update syntax for tensor indexing to new style
import os
import re
import itertools

# list of parameters to adjust

# directory to recursively search for 
root_dir = "."

# file types which are updated
valid_file_endings = [".h", ".cpp"]

# if given, only these files will be updated instead recursively searching for them
# valid_file_endings will be ignored but the path should still be relative to root_dir
files_to_update = []

# Update the syntax for index based access from field[X][Y][Z] to field[{X,Y,Z}].
# Note that this operation does not work on all cases. Since it uses regex,
# sufficiently complex expressions for the indices will be skipped.
# Also, for 4th order tensors like velocity, sometimes a vector is accessed instead
# of a single component. Such cases have to be updated manually, e.g. 
# velocity[X][Y][Z] -> &velocity[{X,Y,Z,0}]
update_tensors = False
# list of 3rd order tensors to update
tensors_3 = ["energy", "energy_previous", 
    "temperature", "previous_temperature", "temp_temperature", 
    "force_thermal", "temp_force_thermal", 
    "solid_thermal_type", "c_p"]
# list of 4th order tensors
tensors_4 = ["pop_t", "pop_old_t"]

# replace exponential expressions like std::pow(x, 2) with more efficient versions
replace_exp = False

# ********************************************************** #
# implementation
def search_and_replace(reg_expr, repl, contents):
	reg_expr = re.compile(reg_expr)
	return re.sub(reg_expr, repl, contents)

def update_tensor_syntax(contents):
	nested_sqr = "[^][]*\[[^][]+\]"
	sqr_brackets = ["\[([^]]+)\]",
		"\[(" + nested_sqr + "[^]]*)\]",
		"\[(" + nested_sqr + nested_sqr + "[^]]*)\]"]
	
	ten_3_rules = []
	for t in itertools.product(sqr_brackets[0:2],sqr_brackets[0:2],sqr_brackets[0:2]):
		ten_3_rules.append("{}{}{}".format(t[0],t[1],t[2]))
	for t in tensors_3:
		# [X][Y][Z] -> [{X,Y,Z}]
		for rule in ten_3_rules: # \[([^]]+)\]\[([^]]+)\]\[([^]]+)\]
			contents = search_and_replace("{}{}".format(t,rule), 
				t + "[{\g<1>,\g<2>,\g<3>}]", contents)

		# double*** t; -> Scalar_field t;
		contents = search_and_replace("double\*\*\* {} = nullptr;".format(t),
			"Scalar_field {};".format(t), contents)

	ten_4_rules = []
	# cartesian product of all variants with 0 or 1 nested brackets
	for t in itertools.product(sqr_brackets[0:2],sqr_brackets[0:2],sqr_brackets[0:2],sqr_brackets[0:2]):
		ten_4_rules.append("{}{}{}{}".format(t[0],t[1],t[2],t[3]))
	ten_4_rules.append("{}{}{}{}".format(sqr_brackets[0],sqr_brackets[2],sqr_brackets[2],sqr_brackets[2]))
	for t in tensors_4:
		# [X][Y][Z][W] -> [{X,Y,Z,W}]
		# {}\[([^]]+)\]\[([^]]+)\]\[([^]]+)\]\[([^]]+)\]
		#reg_expr = re.compile("{}{}{}{}{}".format(t, sqr_brackets[1], sqr_brackets[0], sqr_brackets[0], sqr_brackets[0]))
		for rule in ten_4_rules:
			contents = search_and_replace("{}{}".format(t, rule),
				t + "[{\g<1>,\g<2>,\g<3>,\g<4>}]", contents)

		# double**** t; -> Vector_field t;
		contents = search_and_replace("double\*\*\*\* {} = nullptr;".format(t),
			"Vector_field {};".format(t), contents)

	return contents

def update_sqr(contents):
	first_args = ["[^,]+", "[^[]+\[{[^}]+}\][^,]*"]
	for first_arg in first_args:
		contents = search_and_replace("pow\(({}), 2.0\)".format(first_arg), "sqr(\g<1>)", contents)
		contents = search_and_replace("pow\(({}), 2\)".format(first_arg), "sqr(\g<1>)", contents)
		contents = search_and_replace("pow\(({}), 0.5\)".format(first_arg), "sqrt(\g<1>)", contents)

	return contents

def change_memory_layout(contents):
	tensors_4 = [("pop", (1,2,3,0)), ("pop_old", (1,2,3,0))]
	idx_slot = "([^,]+)"
	idx_slot_end = "([^}]+)"
	groups = ["\g<1>","\g<2>","\g<3>","\g<4>"]
	for t, perm in tensors_4:
		contents = search_and_replace("{}\[\{{{},{},{},{}\}}\]".format(t, idx_slot,idx_slot,idx_slot,idx_slot_end)
			, "{}[{{{},{},{},{}}}]".format(t, groups[perm[0]], groups[perm[1]], groups[perm[2]], groups[perm[3]]), contents)

	return contents


def update_file(file_name):
	contents = ""
	with open(file_name, 'r') as file:
		contents = file.read()
	
	if replace_exp:
		contents = update_sqr(contents)
	if update_tensors:
		contents = update_tensor_syntax(contents)
	#contents = change_memory_layout(contents)
	
	with open(file_name, "w") as file:
		file.write(contents)

if __name__ == '__main__':
	if len(files_to_update) == 0:
		for root, dirs, files in os.walk(root_dir):
			for file in files:
				if os.path.splitext(file)[1] in valid_file_endings:
					files_to_update.append(os.path.join(root,file))
	else:
		files_to_update = [os.path.join(root_dir, file) for file in files_to_update]
	
	for file in files_to_update:
		print("Updating file {}.".format(file))
		update_file(file)