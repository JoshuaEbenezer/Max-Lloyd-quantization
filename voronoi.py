import numpy as np
from pdf_finder import pdf_finder
from scipy.spatial import Voronoi, voronoi_plot_2d
import matplotlib.pyplot as plt
from scipy.spatial import cKDTree
import time

def find_prob(coordinates,test_points,probabilities):	# to find the prob. dens. fn. of the space
	prob_points = []
	for index,[x,y] in enumerate(test_points):
		if (x<coordinates[2]):
			prob_points.append(probabilities[0])
		elif (x<coordinates[3]):
			prob_points.append(probabilities[1])
		else:
			prob_points.append(probabilities[2])	
	return prob_points

def initial_estimate(coordinates,region_number):	# random initial estimate for quantization centroids
	ax = coordinates[5] * (2*np.random.rand(region_number,1)-1) # initial estimates of ax between -n6 and n6
	ay = coordinates[0] * (2*np.random.rand(region_number,1)-1) # initial estimates of ax between -n1 and n1
	a = np.concatenate((ax,ay),axis=1)
	return a

def generate_points(coordinates):	# to return a sampled version of the space
	test_points = []
	for x in [float(j)/10 for j in range(-coordinates[5]*10,coordinates[5]*10)]:
		for y in [float(h)/10 for h in range(-coordinates[4]*10,coordinates[4]*10)]:
			test_points.append([x,y])
	for x in [float(j)/10 for j in range(-coordinates[3]*10,coordinates[3]*10)]:
		for y in [float(h)/10 for h in range(coordinates[4]*10,coordinates[0]*10)]:
			test_points.append([x,y])
		for y in [float(h)/10 for h in range(-coordinates[0]*10,coordinates[4]*10)]:
			test_points.append([x,y])
	return test_points


coordinates = []
print('Enter the numbers n1 to n6 of the joint pdf. Co-ordinates format is (0,n1),(0,n2),(n3,0),(n4,n5),(n6,0) in that order.')
for i in range(6):
	n = input('Enter number ')
	coordinates.append(int(n))
print('Numbers :',coordinates)

ratio1 = float(input('Enter the ratio of p1 and p2 '))
ratio2 = float(input('Enter the ratio of p1 and p3 '))
probabilities = pdf_finder(coordinates,ratio1,ratio2)

print('Probability density functions on the surfaces are ',probabilities)

region_number = int(input('Enter the number of discretization regions '))

estimate = initial_estimate(coordinates,region_number)

print('Initial randomly generated estimates of a in format [ax,ay] :', estimate)

# generate discretized point space within the coordinates
test_points = generate_points(coordinates)
prob_point = find_prob(coordinates,test_points,probabilities)

for i in range(10):

	# construct Voronoi regions from estimates of centroids
	voronoi_kdtree = cKDTree(estimate)
	test_point_dist, test_point_regions = voronoi_kdtree.query(test_points, k=1)

	# test_point_regions now holds an array of shape (n_test, 1)
	# with the indices of the points in voronoi_points closest to each of your test_points.
			
	prob_max = np.zeros(region_number)
	new_estimate = np.zeros((region_number,2))

	for region_index in range(region_number):
		point_indices = np.where(test_point_regions==region_index)
		for point_index in point_indices:
			current_prob = prob_point[point_index[0]][0]
			if (prob_max[region_index]<current_prob):
				prob_max[region_index] = current_prob
				new_estimate[region_index] = test_points[point_index[0]]
	
	error = np.sum(np.square(new_estimate-estimate))
	vor = Voronoi(estimate)
	voronoi_plot_2d(vor)
	
	plt.show(block=False)
	plt.pause(3)
	plt.close()
	estimate = new_estimate
