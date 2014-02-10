#!usr/bin/python3

from numpy import *
from sys import argv
from math import sqrt
from math import isnan
import copy as cop

#za inicijalizaciju centroida koristi se funkcija iz modula kmeans.py

def initialize_pik(k):
	p = []
	for i in range(k):
		p.append(1/k)

	return p
	
def initialize_kovs(k, n):
	m = []
	identity_matrix = identity(n)
	for i in range(k):
		m.append(copy(identity_matrix))

	return m

def gauss_probability(x, mi, kov):
	differ = subtract(x[2], mi[2]) 
	kov_inv = linalg.inv(kov)
	e = dot(differ, kov_inv)
	e = dot(e, differ)
	e = dot(e, -0.5)
	
	d = linalg.det(kov)
	d = d ** 0.5
	p = (2 * pi) ** (len(mi[2]) / 2)
	pe = 1 / (d * p)
	
	return pe * exp(e)

def calculate_L(PIk, centroids, kovs, vehicle):
	L = 0
	for x in vehicle:
		z = 0
		for i, pik in enumerate(PIk):
			z += (pik * gauss_probability(x, centroids[i], kovs[i]))
		L += log(z)
	
	return L

def calculate_group(h):
	maximum = h[0]
	index = 0
	for i, hi in enumerate(h):
		if maximum < hi:
			maximum = hi
			index = i
	
	return index

#vraca listu [broj iteracija potrebnih do konvergencije, L, centroids, listu Lova po iteracijama]
def em_algorithm(centroids, PIk, kovs, vehicle):
	L_previous = None
	L_current = None
	number_of_iterations = 0
	Ls = []
	L = calculate_L(PIk, centroids, kovs, vehicle)
	Ls.append(L)
	
	while True:
		number_of_iterations += 1
		#E-korak
		for x in vehicle:
			h = []
			for i, c in enumerate(centroids):
				probability = gauss_probability(x, c, kovs[i])
				denominator = 0
				for ji, j in enumerate(PIk):
					denominator += (gauss_probability(x, centroids[ji], kovs[ji]) * j)
				h.append(probability * PIk[i] / denominator)
			x[1] = cop.copy(h)
		
		#M-korak

		#odredivanje centroida mi[k]			
		for i, p in enumerate(centroids):
			hi = copy(vehicle[0][2])
			hi = dot(hi, 0)
			hk = 0
			for x in vehicle:
				hk += x[1][i] 
				hi = add(hi, dot(x[1][i], x[2]))
			centroids[i][2] = copy(hi / hk)

		#odredivanje kovarijacijskih matrica
		for i, p in enumerate(kovs):
			hk = 0
			km = None
			for x in vehicle:
				hk += x[1][i]
				differ = subtract(x[2], centroids[i][2])
				m = multiply(transpose([differ]),  differ)
				m = dot(x[1][i], m)
				if km is None:
					km = copy(m)
				else:
					km = add(km, m)
			km = dot(km, 1/hk)
			kovs[i] = copy(km)

		#odredivanje pieva
		h = None
		n = len(vehicle)
		for x in vehicle:
			if h is None:
				h = cop.copy(x[1])
			else:
				h = add(h, x[1])
		for i, hi in enumerate(h):
			PIk[i] = hi / n
	
		#log-izlednost
		L_current = calculate_L(PIk, centroids, kovs, vehicle)
		Ls.append(L)
		if L_previous is not None:
			if abs(L_previous - L_current) < 0.00001:
				break
			
		L_previous = L_current
	
		if isnan(L_previous):
			break
	
	return [number_of_iterations, L_current, centroids, Ls]




