""" Using genetic algorithm with connection encoding to optimize
	a binary feed-forward neural network
"""
import numpy as np


def FNN(w01, w12, a):
	""" Binary FNN with 4 input neurons, 2 interneurons, & 1 output neuron.
		Assume w01, w12 are legal numpy vectors indicating weights.
		The first half of w01 is to the first interneurons, and so on.
	"""
	# from input neurons to interneurons
	w01 = w01.reshape(2, 4)

	return w12 @ (w01 @ a)


def gen(n):
	""" Generate a population of size n.
		Return chromosome 1 encoding w01, and chromosome 2 encoding w12.
	"""
	c1 = np.random.randint(-1, 2, size=(n, 8))
	c2 = np.random.randint(-1, 2, size=(n, 2))

	return c1, c2

def sel(c1, c2, a):
	""" Select the top 25% from the chromosome pools.
		Assume that pool c1 and c2 have the same first dimension >= 8.
		Return survived and extincted cases' indices.
	"""
	n = c1.shape[0]
	fit = np.zeros(n, dtype=np.float64)

	for i in range(n):
		fit[i] = FNN(c1[i], c2[i], a)

	rank = np.argsort(fit) # ranking of fitness
	n = int(n * 0.75) # n now is the index of the first failed case
	survive = rank[n:]
	extinct = rank[:n]

	return survive, extinct

def cross(survive, extinct, c1, c2):
	""" Inplace crossover of parents.
	"""
	k = 0 # counter that helps identify which two parents to crossover next
	n = survive.shape[0] # the number of parents

	for i in extinct:
		p1 = survive[k % n]
		k += 1
		p2 = survive[k % n]

		# even crossover
		c1[i, :4] = c1[p1, :4]
		c1[i, 4:] = c1[p2, 4:]
		c2[i, 0] = c2[p1, 0]
		c2[i, 1] = c2[p2, 1]

def mut(extinct, c1, c2):
	""" Inplace mutation of offsprings.
	"""
	for i in extinct:
		mut_points = np.random.randint(0, 10, size=2)

		for m in mut_points:
			w = np.random.randint(-1, 2)
			if m > 7:
				c2[m % 2] = w
			else:
				c1[m] = w


if __name__ == '__main__':
	n = 12
	c1, c2 = gen(12)
	a = np.random.rand(4)

	for _ in range(1000):
		s, e = sel(c1, c2, a)
		cross(s, e, c1, c2)
		mut(e, c1, c2)

	s, e = sel(c1, c2, a)
	print(c1[s])
	print(c2[s])