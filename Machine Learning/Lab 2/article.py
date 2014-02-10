from numpy import *
from sys import argv

global dimension 
global dictionary

dictionary = []
dimension = 0

rjecnik_file = argv[1]

class Document:
	def __init__(self, y, features):
		self.y = y	
		featuresList = [0 for i in range(dimension)]
		
		for i in range(1, len(features)):
			a = features[i].split(":")
			featuresList[int(a[0])+1] = float(a[1])
		
		featuresList[0] = 1	
		self.features = matrix(featuresList)
			
def read_dictionary(path):
	f = open(path, 'r')
	words = []

	for line in f:
		line = line.rstrip()
		temp = line.split()
		words.append(temp[0])
	
	return words
	
#dictionary = read_dictionary('rjecnik.txt')
dictionary = read_dictionary(rjecnik_file)
dimension = len(dictionary) + 1
