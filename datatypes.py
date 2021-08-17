import numpy as np
from numpy import dot, array, asarray, sqrt
from math import sqrt, pi, sin, cos, radians
from random import uniform
from numpy.linalg import norm
#from numba import njit

#OPERATIONS

class Op:
	def at(ray, dist):
		return ray.origin + (dist*ray.dir)
	#@njit
	def unit(vector):
		return vector/sqrt((vector[0])**2 + (vector[1])**2 + (vector[2])**2)
	
	def urandomacc():
		theta = uniform(0,2*pi)
		z = uniform(-1,1)
		r = sqrt(1 - z*z)
		return np.array((r*cos(theta), r*sin(theta), z), dtype = 'float')
	
	def urandom():
		A = 1/sqrt(3)
		return np.array((uniform(-A,A),uniform(-A,A),uniform(-A,A)), dtype = 'float')
	
	def refract(r,n,e):
		return n*sqrt(1-(e**2)*(1-dot(n,r)**2))+e*(r-dot(n,r)*n)
		'''
		#cos_theta = dot(r, n) #/(norm(r)*norm(n))
		ray_out_L = e*(r + cos_theta*n)
		ray_out_ll = -sqrt(abs(1 - dot(ray_out_L,ray_out_L)))*n
		return ray_out_L + ray_out_ll
		'''
	
	def reflect(r,n):
		return r - 2*dot(r,n)*n
	def schlick(cosine, r_i):
		a = (1-r_i)/(1+r_i)
		b = a*a
		return b + (1 - b)*((1 - cosine)**5)

#VECTOR
class Vector:
	def __init__(self,array):
		self.x = array[0]
		self.y= array[1]
		self.z = array[2]
		self.vec = array

	def unit(self):
		return self.vec/sqrt((self.x)**2 + (self.y)**2 + (self.z)**2)

#RAY
class Ray:
	def __init__(self,og,di):
		def unit(self):
			return self/sqrt((self[0])**2 + (self[1])**2 + (self[2])**2)

		self.origin = np.asarray(og, dtype = float)
		self.originx = og[0]
		self.originy = og[1]
		self.originz = og[2]
		
		self.vec = np.asarray(di, dtype = 'float')
		self.x = self.vec[0]
		self.y = self.vec[1]
		self.z = self.vec[2]
		self.ray = np.array((self.origin, self.vec))

		self.dir = unit(np.asarray(di, dtype = 'float'))
		self.dirx = self.dir[0]
		self.diry = self.dir[1]
		self.dirz = self.dir[2]

#POINT
class Point:
	def __init__(self,array):
		self.x = array[0]
		self.y= array[1]
		self.z = array[2]
		self.pt = array

#COLOR
class Color:
	def __init__(self,array):
		self.x = min(max(0,array[0]),1)
		self.y = min(max(0,array[1]),1)
		self.z = min(max(0,array[2]),1)
		self.col = np.array((self.x, self.y, self.z), dtype ='float')

#CIRCLE
class Circle:
	def __init__(self, array, radius, material):
		self.centre = array.pt
		self.radius = radius
		self.material = material

#MATERIAL
class Material:
	def __init__(self, type, color, info = 0, texture=0):
		self.type = type
		self.color = color
		self.info = info
		self.texture = texture