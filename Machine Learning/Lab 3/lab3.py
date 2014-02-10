#!usr/bin/python3

from numpy import *
from sys import argv
from math import sqrt
import copy as cop
from kmeans import *
from em import *

inputPath = argv[1]
configurationPath = argv[2]
outputDirectoryPath = argv[3]

inputFile = open(inputPath, 'r')
outputFile_em = open(outputDirectoryPath + 'em-all.dat', 'w')
outputFile_kmeans = open(outputDirectoryPath + 'kmeans-all.dat', 'w')
k4out = open(outputDirectoryPath + 'kmeans-k4.dat', 'w')
k4File = open(outputDirectoryPath + 'em-k4.dat', 'w')
confOut = open(outputDirectoryPath + 'em-konf.dat', 'w')
confFile = open(configurationPath, 'r')
emkmeansFile = open(outputDirectoryPath + 'em-kmeans.dat', 'w')

vehicle = []
Ks = [2, 3, 4, 5]

#citanje ulaznih podataka
#jedan primjer: ['naziv', [0,0,0,0,0], [znacajke]]
for line in inputFile:
	line = line.strip()
	features = line.split()
	name = features[-1];
	del features[-1]
	f = array(features, dtype=float32)
	value = []
	value.append(name)
	value.append([0, 0, 0, 0, 0])
	value.append(f)
	vehicle.append(value)

#K-means algoritam	
kmeans_k4_centroids = []

for k in Ks:
	centroids = initialize_centroids(k, vehicle)
	
	result = kmeans_algorithm(centroids, vehicle, k)
	number_of_iterations = result[0]
	j = result[1]
	
	if k == 4:
		kmeans_k4_centroids = cop.copy(result[3])
		k4out.write("#iteracije: J\n--\n")
		for i, j4 in enumerate(result[2]):
			k4out.write("#%d: %.2f\n" % (j4[0], j4[1]))
		k4out.write("--\n")
		for i in range(4):
			group = [0,0,0,0,0]
			group[i] = 1 
			d = {
				'bus' : 0,
				'van' : 0,
				'opel' : 0,
				'saab' : 0
			}
			for x in vehicle:
				if dot(x[1], group) == 1:
					d[x[0]] += 1
					
			k4out.write("Grupa %d: " % (i+1))
			s = sorted(d, key=d.get)
			s.reverse()
			cc = 0
			for car in s:
				cc += 1
				k4out.write(car)
				if cc == 4:
					k4out.write(" %d" % d[car])
				else:
					k4out.write(" %d, " % d[car])
			k4out.write("\n")
	
	#odredivanje broja primjera u pojedinoj grupi
	groups = [0,0,0,0,0]
	for x in vehicle:
		groups = add(groups, x[1])
	
	#ispis u datoteku
	outputFile_kmeans.write("K = " + str(k) + "\n")
	counter = 0
	for c in centroids:
		counter += 1
		outputFile_kmeans.write("c" + str(counter) + ":")
		for z in c[2]:
			outputFile_kmeans.write(" %.2f" % z)
		outputFile_kmeans.write("\ngrupa %d: %d primjera\n" % (counter, groups[counter-1]))
	outputFile_kmeans.write("#iter: %d\n" % number_of_iterations)
	outputFile_kmeans.write("J: %.2f\n--\n" % j)

#EM algoritam

##pokretanje EM algoritma za razliciti broj grupa
for k in Ks:
	centroids = initialize_centroids(k, vehicle)
	PIk = initialize_pik(k)
	kovs = initialize_kovs(k, len(centroids[0][2]))			
				
	result = em_algorithm(centroids, PIk, kovs, vehicle)
	number_of_iterations = result[0]
	L = result[1]
	centroids = result[2]
	
	#ispis svih primjera za K=4
	if k == 4:
		for i in range(4):
			k4File.write("Grupa %d:\n" % (i+1))
			d = []
			for x in vehicle:
				group = calculate_group(x[1])
				if group == i:
					 d.append((x[0], x[1][group]))
			d = sorted(d, key=lambda x: x[1])
			d.reverse()
			for el in d:
				k4File.write("%s %.2f\n" % (el[0], el[1]))
			if i != 3:
				k4File.write("--")
			k4File.write("\n")
	
	#brojanje koliko primjera ima u kojoj grupi
	groups = [0, 0, 0, 0, 0]
	for x in vehicle:
		groups[calculate_group(x[1])] += 1
	
	#ispis u datoteku
	outputFile_em.write("K = %d\n" % k)
	counter = 0
	for c in centroids:
		counter += 1
		outputFile_em.write("c" + str(counter) + ":")
		for z in c[2]:
			outputFile_em.write(" %.2f" % z)
		outputFile_em.write("\ngrupa %d: %d primjera\n" % (counter, groups[counter-1]))
	outputFile_em.write("#iter: %d\n" % number_of_iterations)
	outputFile_em.write("log-izglednost: %.2f\n--\n" % L)

#dodatno ispitivanje ovisnosti pocetnih centroida o ishodu grupiranja

#ucitavanje 5 konfiguracija pocetnih centroida
confs = []
conf1 = []
for i, line in enumerate(confFile):
	line = line.strip()
	features = line.split()
	if len(features) < 3:
		if len(conf1) > 0:
			confs.append(cop.copy(conf1))
			conf1 = []
		continue
	name = features[-1];
	del features[-1]
	f = array(features, dtype=float32)
	value = []
	value.append(name)
	value.append([0, 0, 0, 0, 0])
	value.append(f)
	conf1.append(value)


if len(conf1) != 0:
	confs.append(cop.copy(conf1))
	
for conf_nb, centroids in enumerate(confs):
	PIk = initialize_pik(4)
	kovs = initialize_kovs(4, len(centroids[0][2]))			
		
	result = em_algorithm(centroids, PIk, kovs, vehicle)
	number_of_iterations = result[0]
	L = result[1]
	centroids = result[2]
	
	confOut.write("Konfiguracija %d:\n" % (conf_nb + 1))
	confOut.write("log-izglednost: %.2f\n" % L)
	confOut.write("#iteracija: %d\n" % number_of_iterations)
	if conf_nb != 4:
		confOut.write("--\n")		
				

#pokretanje EM algoritma za K=4 uz pocetne centroide dobivene k-means algoritmom

PIk = initialize_pik(4)
kovs = initialize_kovs(4, len(centroids[0][2]))			

result = em_algorithm(kmeans_k4_centroids, PIk, kovs, vehicle)
number_of_iterations = result[0]
L = result[1]
centroids = result[2]			
Ls = result[3]
				
emkmeansFile.write("#iteracije: log-izglednost\n--\n")
for i, l in enumerate(Ls):				
	emkmeansFile.write("#%d: %.2f\n" % (i, l))
emkmeansFile.write("--\n")


for i in range(4):
	group = [0,0,0,0,0]
	group[i] = 1 
	d = {
		'bus' : 0,
		'van' : 0,
		'opel' : 0,
		'saab' : 0
	}
	for x in vehicle:
		if dot(x[1], group) == 1:
			d[x[0]] += 1
			
	emkmeansFile.write("Grupa %d: " % (i+1))
	s = sorted(d, key=d.get)
	s.reverse()
	cc = 0
	for car in s:
		cc += 1
		emkmeansFile.write(car)
		if cc == 4:
			emkmeansFile.write(" %d" % d[car])
		else:
			emkmeansFile.write(" %d, " % d[car])
	emkmeansFile.write("\n")


