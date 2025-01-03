import numpy as np
import scipy

h = 0.01

def calculate_derivative(f, i: int, x: np.array):
	"""
		f must take a vector of the same size as x and return a real number
		i must be between 0 and size(x)-1
	"""
	x_a, x_b = x.copy(), x.copy()
	x_a[i] -= h
	x_b[i] += h
	return (f(x_b) - f(x_a)) / (2*h)

def calculate_order_2_derivative(f, i: int, j: int, x: np.array):
	"""
		f must take a vector of the same size as x and return a real number
		i and j must be between 0 and size(x)-1
	"""
	a, b, d, c = x.copy(), x.copy(), x.copy(), x.copy()
	a[i] += h; a[j] +=h
	b[i] += h; b[j] -=h
	c[i] -= h; c[j] +=h
	d[i] -= h; d[j] -=h

	return (1/(h**2)) * (f(a) - f(b) - f(c) + f(d))

def calculate_gradient(f, x):
	"""
		f must take a vector of the same size as x and return a real number
	"""
	grad = np.zeros(x.shape)
	for i in range(x.shape[0]):
		grad[i, 0] = calculate_derivative(f, i, x)
	return grad

def calculate_jacobian(f, x):
	"""
		f must take a vector of the same size as x and return a vector of any size
	"""
	jacob = np.zeros((f(x).shape[0], x.shape[0]))
	for i in range(jacob.shape[0]):
		f_i = lambda x: f(x)[i]
		for j in range(jacob.shape[1]):
			jacob[i,j] = calculate_derivative(f_i, j, x)
	return jacob

def calculate_hessian(f, x):
	hess = np.zeros((x.shape[0], x.shape[0]))
	for i in range(hess.shape[0]):
		for j in range(hess.shape[1]):
			hess[i,j] = calculate_order_2_derivative(f, i, j, x)
	return hess

def sqp_algorithm(f, g, x0, l0):
	def sqp_step(x, lamb):
		l = lambda y: f(y) + (np.transpose(lamb) @ g(y))[0,0]
		lagrange_hessian = calculate_hessian(l, x)

		def objective_function(a: np.array):
			a_vec = np.transpose(np.array([a]))
			return (np.transpose(a_vec)@((0.5*lagrange_hessian)@a_vec + calculate_gradient(f, x))).item()
	
		def constrain_function (a): 
			a_vec = np.transpose(np.array([a]))
			return (g(x) + calculate_jacobian(g, x)@a_vec).flatten()
		
		constrain_lower_bound = np.zeros(lamb.shape[0])
		constrain_upper_bound = constrain_lower_bound

		problem_constrains = scipy.optimize.NonlinearConstraint(
			constrain_function,
			constrain_lower_bound,
			constrain_upper_bound
		)

		initial_point = np.random.random(x.shape[0])*4

		result = scipy.optimize.minimize(
			objective_function,
			initial_point,
			constraints=problem_constrains,
			method='trust-constr'
		)
		return np.transpose(np.array([result.x])), np.transpose(np.array([result.v[0]]))
	
	x_current, l_current = x0, l0
	for _ in range(1):
		x_current, l_current = sqp_step(x_current, l_current)

	return x_current, l_current

def newton_method(f, g, x0, l0, num_iters=100):
	def newton_method_step(x, lamb):
		l = lambda y: f(y) + (np.transpose(lamb) @ g(y))[0,0]

		Q_tl = calculate_hessian(l, x)
		Q_bl = calculate_jacobian(g, x)
		Q_tr = np.transpose(Q_bl.copy())
		Q_br = np.zeros((lamb.shape[0], lamb.shape[0]))
		q = np.block([[Q_tl, Q_tr], [Q_bl, Q_br]])

		right_top = calculate_gradient(f, x) + np.transpose(np.transpose(lamb)@calculate_jacobian(g, x))
		right_bottom = g(x)
		right = np.concatenate([right_top, right_bottom], 0)

		res = -np.linalg.inv(q) @ right
		return (x + res[:x.shape[0], [0]], lamb + res[x.shape[0]:, [0]])
	
	x_current, l_current = x0, l0
	for _ in range(num_iters):
		x_current, l_current = newton_method_step(x_current, l_current)

	return x_current, l_current