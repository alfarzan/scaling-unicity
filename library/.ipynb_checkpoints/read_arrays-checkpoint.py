import datetime
from functools import partial
import os
import numpy as np
from math import gcd
import multiprocessing as mp
import time
from threading import Thread

def get_flist(nusers = int(4e6)):
	flist = []
	numlist = list(range(10))
	print('Getting file names:\n')
	breaker = False
	for i in numlist:
		if breaker:
			break
		for j in numlist:
			if breaker:
				break
			for k in numlist:
				diradd = '/scratch/PrivacySignature/users/{}/{}/{}/'.format(k, j, i)
				flist += list(map(lambda x:diradd+x, os.listdir(diradd)))
				if len(flist) > nusers:
					breaker = True
					break
	return flist[:nusers]

def get_date_array(start_date = '2006-03-01', end_date = '2006-06-01'):
	start = datetime.datetime.strptime(start_date, '%Y-%m-%d')
	end = datetime.datetime.strptime(end_date, '%Y-%m-%d')
	step = datetime.timedelta(seconds = 3600)
	current = start
	ddict = []
	while current < end:
		ddict.append(current.strftime('%Y-%m-%d %H'))
		current += step
	return np.array(ddict)

def get_ant_array():
	diradd = '/scratch/prince_ali/scaling/PT_processed.txt'
	adict = []
	with open(diradd, 'r') as fadd:
		for line in fadd:
			ant = line.strip().split(' ')[0]
			adict.append(ant)
	return np.array(adict)


def get_rand_iterator(itsize, rand_seed = 42, max_ties = 43):
	np.random.seed(rand_seed)
	def lcm(a,b):
		if a != 0 and b != 0:
			return int(abs(a * b)/gcd(a,b))
		return 0
	target_lcm = 1
	for i in range(1,max_ties):
		target_lcm = lcm(target_lcm, i)
	return  iter(np.random.randint(target_lcm, size = itsize))


def get_user_track(ufile, date_array, ant_array, rand_ind):
	
	with open(ufile, 'r') as myf:
		inds = numbarator(myf, date_array, ant_array, rand_ind)
	return inds

	

def read_lines(line, date_array, ant_array):
	timestamp = line[:13]
	antenna = line[20:-1]
	if timestamp in date_array:
		timestamp = np.where(date_array == timestamp)[0][0]
		antenna = np.where(ant_array == antenna)[0][0]
		return (timestamp, antenna)
	else:
		return (-1,-1)

# def winner_takes_all(choices, rand_ind):
# 	if len(choices) == 1:
# 		return choices[0]
# 	unique_choices = np.zeros(len(choices), dtype = np.int64) - 1 
# 	counts = np.zeros(len(choices), dtype = np.int64) - 1
# 	counter = 0
# 	for ind in range(len(choices)):
# 		if choices[ind] in unique_choices:
# 			counts[np.where(unique_choices == choices[ind])] += 1
# 		else:
# 			unique_choices[counter] = choices[ind]
# 			counts[counter] = 1
# 			counter += 1

# 	counts = counts[:len(unique_choices)]
# 	unique_choices = unique_choices[:len(unique_choices)]
# 	sorter = np.argsort(-counts)
# 	counts = counts[sorter]
# 	unique_choices = unique_choices[sorter]

# 	equal_list = []
# 	prevcount = counts[0]
# 	for ind in range(len(counts)):
# 		if counts[ind] == prevcount:
# 			equal_list.append(ind)
# 		else:
# 			break
# 	unique_choices = unique_choices[equal_list]
# 	del equal_list
# 	del counts
# 	del sorter
# 	return unique_choices[rand_ind%len(unique_choices)]


def winner_takes_all(choices):
	if len(choices) == 1:
		return choices[0]
	unique_choices = np.zeros(len(choices), dtype = np.int64) - 1 
	counts = np.zeros(len(choices), dtype = np.int64) - 1
	counter = 0
	for ind in range(len(choices)):
		if choices[ind] in unique_choices:
			counts[np.where(unique_choices == choices[ind])] += 1
		else:
			unique_choices[counter] = choices[ind]
			counts[counter] = 1
			counter += 1

	counts = counts[:len(unique_choices)]
	unique_choices = unique_choices[:len(unique_choices)]
	sorter = np.argsort(-counts)
	counts = counts[sorter]
	unique_choices = unique_choices[sorter]

	equal_list = []
	prevcount = counts[0]
	for ind in range(len(counts)):
		if counts[ind] == prevcount:
			equal_list.append(ind)
		else:
			break
	unique_choices = unique_choices[equal_list]
	del equal_list
	del counts
	del sorter
	return np.random.choice(unique_choices)



def numbarator(myf, date_array, ant_array):

	a = []
	t = []
	for line in myf:
		point = read_lines(line, date_array = date_array, ant_array = ant_array)
		if point != (-1, -1):
			a.append(point[1])
			t.append(point[0]) 
	if len(t) == 0:
		return []
	times = np.array(t)
	ants = np.array(a)
	prime_sorter = np.argsort(times)
	times = times[prime_sorter]
	ants = ants[prime_sorter]
	track = list(zip(times, ants))
	track.append((0,0))
	prevtime = times[0]
	prevant = ants[0]
	choices = [prevant]
	newtrack = []
	for time, ant in track:
		if time == prevtime:
			choices.append(ant)
		else:
			picked_ant = winner_takes_all(choices)
			choices = [ant]
			newtrack.append((prevtime, picked_ant))			
		prevtime = time
		prevant = ant

	total_hrs = len(date_array)
	total_ants = len(ant_array)
	inds = []
	for point in newtrack:
		i = point[0]
		j = point[1]
		inds.append(total_ants*i+j)

	del times
	del ants
	del newtrack
	del track
	del choices
	del prime_sorter
	return inds

# def numbarator(myf, date_array, ant_array, rand_ind):

# 	a = []
# 	t = []
# 	for line in myf:
# 		point = read_lines(line, date_array = date_array, ant_array = ant_array)
# 		if point != (-1, -1):
# 			a.append(point[1])
# 			t.append(point[0])
# 	if len(t) == 0:
# 		return []
# 	times = np.array(t)
# 	ants = np.array(a)
# 	prime_sorter = np.argsort(times)
# 	times = times[prime_sorter]
# 	ants = ants[prime_sorter]
# 	track = list(zip(times, ants))
# 	track.append((0,0))
# 	prevtime = times[0]
# 	prevant = ants[0]
# 	choices = [prevant]
# 	newtrack = []
# 	for time, ant in track:
# 		if time == prevtime:
# 			choices.append(ant)
# 		else:
# 			picked_ant = winner_takes_all(choices, next(rand_ind))
# 			choices = [ant]
# 			newtrack.append((prevtime, picked_ant))			
# 		prevtime = time
# 		prevant = ant

# 	total_hrs = len(date_array)
# 	total_ants = len(ant_array)
# 	inds = []
# 	for point in newtrack:
# 		i = point[0]
# 		j = point[1]
# 		inds.append(total_ants*i+j)

# 	del times
# 	del ants
# 	del newtrack
# 	del track
# 	del choices
# 	del prime_sorter
# 	del rand_ind
# 	return inds

# def get_and_save_user_indices(ufile, date_array, ant_array, rand_ind, root):
	
# 	with open(ufile, 'r') as myf:
# 		inds = numbarator(myf, date_array, ant_array, rand_ind)
# 	if len(inds) != 0:
# 		np.save(root+ufile[32:46], inds)
# 	del inds
# 	del rand_ind

def get_and_save_user_indices(ufile, date_array, ant_array, root):
	
	with open(ufile, 'r') as myf:
		inds = numbarator(myf, date_array, ant_array)
	if len(inds) != 0:
		np.save(root+ufile[32:46], inds)
	del inds




def get_bool_array(indices, date_len, ant_len):
	utrack = np.zeros(ant_len*date_len, dtype = (bool))
	utrack[indices] = True
	return utrack

def create_directory_structure(start_date = '2006-03-01', end_date = '2006-06-01'):
	sdate, edate = ''.join(start_date.split('-')), ''.join(end_date.split('-'))
	root = '/scratch/prince_ali/scaling/user_numpy_arrays/'+sdate+'_'+edate+'/'
	if not os.path.exists(root):
		os.makedirs(root)
	numrange = list(range(10))
	print('Creating the directory structure')
	for i in numrange:
		diradd = root+'{}/'.format(i)
		if not os.path.exists(diradd):
			os.makedirs(diradd)
		for j in numrange:
			diradd = root+'{}/{}/'.format(i,j)
			if not os.path.exists(diradd):
				os.makedirs(diradd)
			for k in numrange:
				diradd = root+'{}/{}/{}/'.format(i,j,k)
				if not os.path.exists(diradd):
					os.makedirs(diradd)
	return root


def chunckit(a, n):
	ave_len = len(a)//n
	last_len  = ave_len + len(a)%n
	groups = []
	for i in range(n):
		if i == n-1:
			groups.append(a[i*ave_len:])
			break
		groups.append(a[i*ave_len:(i+1)*ave_len])
	return groups


def numpy_saver(index, track_array, name_array):
	np.save(name_array[index], track_array[index])
	


def build_arrays(flist, ant_array, date_array, root):
	rand_seed = int(flist[0][38:46])
	# rand_ind = get_rand_iterator(len(date_array)*len(flist), rand_seed = rand_seed)
	# my_track_extractor = partial(get_user_track, date_dict = date_dict, ant_dict = ant_dict, rand_ind = rand_ind)


	# def my_track_saver(fname, date_array = date_array, ant_array = ant_array, rand_ind = rand_ind, root = root):
	# 	return get_and_save_user_indices(fname, date_array, ant_array, rand_ind, root)

	my_track_saver = partial(get_and_save_user_indices, date_array = date_array, ant_array = ant_array, root = root)
	
	print('Starting array computation. Seed: {}\n'.format(rand_seed))
	start = time.time()
	for fname in flist:
		my_track_saver(fname)
	print('Seed: {} took {} seconds.\n=====================\n'.format(rand_seed, time.time()-start))


def split_the_work(start_date = '2006-03-01', end_date = '2006-06-01', nproc = 12, nusers = int(4e6)):
	flist = get_flist(nusers)
	flist_groups = chunckit(flist, nproc)
	del flist
	date_array = get_date_array(start_date, end_date)
	ant_array = get_ant_array()
	root = create_directory_structure(start_date, end_date)


	map_target = partial(build_arrays, ant_array = ant_array, date_array = date_array, root = root)


	# allusers = sum(list(map(len, flist_groups)))
	# start = time.time()
	# n = 200
	# while len(flist_groups) != 0:
	# 	map_target(flist_groups[0][:n])
	# 	del flist_groups[0]
	# end = time.time()
	# dur = end-start
	# print('For {} users this took {} seconds. For {} people it will take about {} hours'.format(n*nproc, dur, allusers, allusers*dur/(n*nproc)/3600))

	# threads = []
	# print('Starting threads:\n')
	# count = 0
	# while len(flist_groups) != 0:
	# 	print('starting thread {}\n'.format(count))
	# 	count += 1
	# 	t = Thread(target = map_target, args = (flist_groups[0],))
	# 	del flist_groups[0]
	# 	threads.append(t)
	# 	t.start()
		
	# print('Joining {} threads./n'.format(count))
	# for t in threads:
	# 	t.join()
	
	mymap = mp.Pool(processes = nproc)
	mymap.map(map_target, flist_groups)
	print('Joining the processes.\n')
	
	
	

# #====================test to confirm sparse representation is correct=========================

# start, end = '2006-03-01', '2006-06-01'

# date_array = get_date_array(start_date = start, end_date = end)

# ant_array = get_ant_array()


# rand_ind = get_rand_iterator(int(3e6))

# flist = get_flist()
# ufile = flist[1000]

# inds = get_user_track(ufile, date_array, ant_array, rand_ind)

# tot_hrs, tot_ants = len(date_array), len(ant_array)

# utrack = [(i//tot_ants, i%tot_ants)  for i in inds]
# utrack = [(date_array[i], ant_array[j]) for i,j in utrack]

# import pandas as pd
# sdate = datetime.datetime.strptime('2006-03-01','%Y-%m-%d')
# edate = datetime.datetime.strptime('2006-06-01','%Y-%m-%d')

# real_df = pd.read_csv(ufile, names = ['date', 'antenna'])
# real_df['date'] = pd.to_datetime(real_df['date'], format = '%Y-%m-%d %H:%M:%S')

# real_df = real_df[real_df['date'] >= sdate]
# real_df = real_df[real_df['date'] <= edate]

# my_df = pd.DataFrame(utrack, columns = ['date', 'antenna'])


# #=========================test to confirm numpy arrays are being saved==================
# build_arrays(nusers = 10)

#===================test to confirm chunkit wokrs properly=========================
# flist = get_flist()

# groups = chunckit(flist, 12)

#========================testing the paralellism==========================================
start = time.time()
split_the_work(nproc = 18)
dur = time.time()-start
print('The whole thing took about {} hrs'.format(dur/3600))
# #====================test to confirm sparse representation is correct=========================

# start, end = '2006-03-01', '2006-06-01'

# date_dict = get_date_dict(start_date = start, end_date = end)
# int_dict = {j:i for i,j in date_dict.items()}
# ant_dict = get_ant_dict()
# int_ant_dict = {j:i for i, j in ant_dict.items()}

# uid = 24600000
# booltrack = np.load('/scratch/prince_ali/scaling/user_numpy_arrays/20060301_20060601/0/0/0/{}.npy'.format(uid))

# tot_hrs, tot_ants = len(date_dict), len(ant_dict)
# inds = np.nonzero(booltrack)[0]
# utrack = [(i//tot_ants, i%tot_ants)  for i in inds]
# utrack = [(int_dict[i], int_ant_dict[j]) for i,j in utrack]

# import pandas as pd
# sdate = datetime.datetime.strptime('2006-03-01','%Y-%m-%d')
# edate = datetime.datetime.strptime('2006-06-01','%Y-%m-%d')

# real_df = pd.read_csv('/scratch/PrivacySignature/users/0/0/0/{}.csv'.format(uid), names = ['date', 'antenna'])
# real_df['date'] = pd.to_datetime(real_df['date'], format = '%Y-%m-%d %H:%M:%S')

# real_df = real_df[real_df['date'] >= sdate]
# real_df = real_df[real_df['date'] <= edate]

# my_df = pd.DataFrame(utrack, columns = ['date', 'antenna'])
