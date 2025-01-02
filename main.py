from math import exp, cos, sin
from newton_method import *
import numpy as np

def test_f(x):
	""" Function to be optimize using the SQP algorithm """
	assert x.shape == (5,1)
	product = 1
	for v in x:
		product *= v.item()
	return exp(product) - 0.5*((x[0][0]**2+x[1][0]**2+1)**2)

def test_g(x):
	assert x.shape == (5,1)
	return np.transpose(np.array([[
		(np.transpose(x)@x - 10)[0,0],
		x[1,0]*x[2,0] - 5*x[3,0]*x[4,0],
		1 + x[0,0]**3 + x[1,0]**3
	]]))

def test():
	error = sum([abs(calculate_derivative(np.cos, 0, [value]) + sin(value)) for value in np.arange(0.0, 6.28, 0.3)])
	print(error)

if __name__ == "__main__":
	# Example functions given by the ENSTA course
	x_0 = np.array([[-1.71, 1.59, 1.82, -0.763, -0.763]]).transpose()
	l_0 = np.ones((3,1))
	print(newton_method(test_f, test_g, x_0, l_0))
	print(sqp_algorithm(test_f, test_g, x_0, l_0))

	# Trivial example to test the 1D case
	# print(newton_method(
	# 	lambda x: np.array([[x[0,0]*x[0,0]]]),
	# 	lambda x: np.array([[x[1,0]]]),
	# 	np.array([[4.0], [3.0]]),
	# 	np.array([[4.0]]),
	# 	100
	# ))