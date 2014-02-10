#!usr/bin/python3

from numpy import *
from sys import argv
from math import sqrt
import copy as cop

#inicijalizacija cenroida
def initialize_centroids(K, vehicle):
	centroids1 = []
	if K == 2:
		vehicle[4][1] = [1,0,0,0,0]
		centroids1.append(cop.copy(vehicle[4]))
		vehicle[15][1] = [0,1,0,0,0]
		centroids1.append(cop.copy(vehicle[15]))
	
	if K == 3:
		vehicle[0][1] = [1,0,0,0,0]
		centroids1.append(cop.copy(vehicle[0]))
		vehicle[4][1] = [0,1,0,0,0]
		centroids1.append(cop.copy(vehicle[4]))
		vehicle[15][1] = [0,0,1,0,0]
		centroids1.append(cop.copy(vehicle[15]))
		
	if K == 4:
		vehicle[0][1] = [1,0,0,0,0]
		centroids1.append(cop.copy(vehicle[0]))
		vehicle[2][1] = [0,1,0,0,0]
		centroids1.append(cop.copy(vehicle[2]))
		vehicle[4][1] = [0,0,1,0,0]
		centroids1.append(cop.copy(vehicle[4]))
		vehicle[15][1] = [0,0,0,1,0]
		centroids1.append(cop.copy(vehicle[15]))
	
	if K == 5:
		vehicle[0][1] = [1,0,0,0,0]
		centroids1.append(cop.copy(vehicle[0]))
		vehicle[2][1] = [0,1,0,0,0]
		centroids1.append(cop.copy(vehicle[2]))
		vehicle[4][1] = [0,0,1,0,0]
		centroids1.append(cop.copy(vehicle[4]))
		vehicle[9][1] = [0,0,0,1,0]
		centroids1.append(cop.copy(vehicle[9]))
		vehicle[15][1] = [0,0,0,0,1]
		centroids1.append(cop.copy(vehicle[15]))
	
	return centroids1

#norm = ||x-mi||^2
def calculate_norm(a, b):
	differ = subtract(a, b)
	return dot(differ, differ)
	
	
def calculate_class_kmeans(example, centroids):
	minimum = 0
	group = -1
	counter = 0
	for centroid in centroids:
		distance = sqrt(calculate_norm(example[2], centroid[2]))
		if group == -1:
			group = counter
			minimum = distance
		elif distance < minimum:
			group = counter
			minimum = distance
		counter += 1
	
	g = [0, 0, 0, 0, 0]
	g[group] = 1
	
	return g

def isChanged(m_start, m_end):
	return array_equal(m_start, m_end)
	
def calculate_J(centroids, vehicle):
	error = 0
	for x in vehicle:
		for centroid in centroids:
			b = dot(x[1], centroid[1])
			norm = calculate_norm(x[2], centroid[2])
			error += (b * norm)

	return error

#vraca listu [broj iteracija, J, listu Jeva u svakom koraku za k=4, centroidi]
def kmeans_algorithm(centroids, vehicle, k):
	number_of_iterations = -1
	Jks = []
	while True:
		number_of_iterations += 1
		done = True
		
		#odredivanje klasa za svaki primjer
		for x in vehicle:
			b = calculate_class_kmeans(x, centroids)
			x[1] = copy(b)
		
		if k == 4:
			jk = calculate_J(centroids, vehicle)
			Jks.append((number_of_iterations, jk))	
		
		#osvjezavanje centroida
		for c in centroids:
			new_centroid = copy(vehicle[0][2])
			new_centroid = dot(0, new_centroid)
			number_of_examples = 0
			for x in vehicle:
				new_centroid = add(dot(dot(c[1], x[1]), x[2]), new_centroid)
				if dot(c[1], x[1]) == 1:
					number_of_examples += 1
			
			if number_of_examples != 0:
				new_centroid = dot(new_centroid, 1/number_of_examples)
			
			#provjera ako je centroid ostao isti(ako barem jedan nije ostao isti onda se ulazi ponovno u while)
			if not isChanged(new_centroid, c[2]):
				done = False
	
			c[2] = copy(new_centroid)

		if done:
			break
	
	j = calculate_J(centroids, vehicle)
	
	number_of_iterations += 1
	if k == 4:
		return [number_of_iterations, j, Jks, centroids]
	return [number_of_iterations, j]
				
				
				
				
				
				 
		

