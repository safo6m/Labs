#!usr/bin/python3

from numpy import *
from math import *
from article import *
from sys import argv

training_file = argv[2] 
validation_file = argv[3]
test_file = argv[4]
output_folder = argv[5]
tezine1_file = output_folder + 'tezine1.dat'
tezine2_file = output_folder + 'tezine2.dat'
optimizacija_file = output_folder + 'optimizacija.dat'
ispitni_predikcije_file = output_folder + 'ispitni_predikcije.dat'
rijeci_file = output_folder + 'rijeci.txt'

training_set = []
validation_set = []
test_set = []

def read_file(path):
	f = open(path, 'r')
	articles = []

	for line in f:
		line = line.rstrip()
		temp = line.split()
		
		doc = Document(int(temp[0]), temp)
		
		articles.append(doc)
	
	return articles
	
def h(w, x):
	a = w * x.transpose()
	return 1/(1+exp(-a))
	
def E(s, w, lam):
	result = 0
	for i in range(len(s)):
		if s[i].y == 0:
			result = result + log( 1- h(w, s[i].features) )
		else:
			result = result + log( h(w, s[i].features) ) 
			
	result = result * (-1)
	w_temp = w.copy()
	w_temp[0, 0] = 0
	result = result + lam/2 * (w_temp * w_temp.transpose())
	
	return result
	
def line_search(s, delta_w, w, lam):
	eta = 0
	delta_eta = 0.01
	e_previous = E(s, w, lam)
	
	while(True):
		eta = eta + delta_eta
		if eta > 1:
			return 1
		new_w = w*(1-eta*lam) - (eta * delta_w)
		#new_w[0, 0] = w[0, 0] - (eta * delta_w[0, 0])
		
		e = E(s, new_w, lam)
		
		if e_previous < e:
			return eta - delta_eta
		
		e_previous = E(s, new_w, lam)
		
def batch_gradient_descent(s, lam):
	w = matrix([0 for i in range(dimension)])
	counter = 0
	
	while(True):
		counter += 1
		e_previous = E(s, w, lam)
		delta_w = matrix([0 for i in range(dimension)])
		for i in range(len(s)):
			m = h(w, s[i].features)
			delta_w = delta_w + (m - s[i].y ) * s[i].features
		eta = line_search(s, delta_w, w, lam)
		w_temp = w.copy()
		w = w*(1-eta*lam) - eta*delta_w
		w[0, 0] = w_temp[0, 0] - eta*delta_w[0, 0]
		
		e = E(s, w, lam)
		print("Iteration: " + str(counter) + "   eta: " + str(eta) + "   E(w|D)=" + str(e))
		
		if abs(e-e_previous) < 0.001:
			return w
	

def cross_validation():
	lambdas = [0, 0.1, 1, 5, 10, 100, 1000]
	gen_faults = []
	optimal_lambda_index = 0
	index = -1

	for lam in lambdas:
		index += 1
		total = 0
		wrong_classified = 0
		w = batch_gradient_descent(training_set, lam)
		
		for i in range(len(validation_set)):
			total += 1
			classification = h(w, validation_set[i].features)
			if classification >= 0.5:
				classification = 1
			else:
				classification = 0
			
			if classification != validation_set[i].y:
				wrong_classified += 1
		
		gen_fault = wrong_classified/total
		gen_faults.append(gen_fault)
		
		if gen_fault <= gen_faults[optimal_lambda_index]:
			optimal_lambda_index = index
		
		print("Lambda: " + str(lam))
		print("gen pogre: " + str(gen_fault))
		print(E(training_set, w, lam)[0,0])
		print(E(validation_set, w, lam))
	
	#f = open('optimizacija.dat', 'w')
	f = open(optimizacija_file, 'w')
	for i in range(len(lambdas)):
		f.write("L=" + str(lambdas[i]) + ", " + str(gen_faults[i]) + "\n")
		print("lambda: " + str(lambdas[i]) + " val: " + str(gen_faults[i]))
	
	f.write("optimalno: L=" + str(lambdas[optimal_lambda_index]) + "\n")
	print("Optimal lambda: " + str(lambdas[optimal_lambda_index]))
	
	return lambdas[optimal_lambda_index]
	
def find_max1(w):
	max_element = w.item(0)
	max_index = 0
	for i in range(w.size):
		if w.item(i) > max_element:
			max_element = w.item(i)
			max_index = i
	return max_index

training_set = read_file(training_file)
validation_set = read_file(validation_file)
test_set = read_file(test_file)

w = batch_gradient_descent(training_set, 0)
#f = open('tezine1.dat', 'w')
f = open(tezine1_file, 'w')
for i in range(w.size):
	f.write("%.2f\n" % w.item(i))
total = 0
wrong_classified = 0
for i in range(len(training_set)):
	total += 1
	classification = h(w, training_set[i].features)
	if classification >= 0.5:
		classification = 1
	else:
		classification = 0
	
	if classification != training_set[i].y:
		wrong_classified += 1

f.write("EE: %.2f\n" % (wrong_classified/total))
f.write("CEE: %.2f\n" % E(training_set, w, 0)[0, 0])

#CrossValidation
optimal_lambda = cross_validation()

big_set = training_set + validation_set
w = batch_gradient_descent(big_set, optimal_lambda)

#f = open('tezine2.dat', 'w')
f = open(tezine2_file, 'w')
for i in range(w.size):
	f.write("%.2f\n" % w.item(i))
total = 0
wrong_classified = 0
for i in range(len(big_set)):
	total += 1
	classification = h(w, big_set[i].features)
	if classification >= 0.5:
		classification = 1
	else:
		classification = 0
	
	if classification != big_set[i].y:
		wrong_classified += 1

f.write("EE: %.2f\n" % (wrong_classified/total))
f.write("CEE: %.2f\n" % E(big_set, w, optimal_lambda)[0, 0])

#20 najboljih rijeci
w_temp = w.copy()
indexes = []

#f = open('rijeci.txt', 'w')
f = open(rijeci_file, 'w')
for i in range(20):
	index = find_max1(w_temp)
	indexes.append(index-1)
	w_temp[0, index] = -1

for i in indexes:
	f.write(str(dictionary[i]) + "\n")
	print(str(i) + ". " + str(dictionary[i]) + " -- " + str(w.item(i)))

#klasifikacija ispitnog skupa
#f = open('ispitni-predikcije.dat', 'w')
f = open(ispitni_predikcije_file, 'w')
total = 0
wrong_classified = 0
for i in range(len(test_set)):
	total += 1
	classification = h(w, test_set[i].features)
	if classification >= 0.5:
		classification = 1
	else:
		classification = 0
	
	if classification != test_set[i].y:
		wrong_classified += 1

	f.write(str(classification) + "\n")
	print(classification)

f.write("Greška: " + str(wrong_classified/total))
print("greska: " + str(wrong_classified/total))

