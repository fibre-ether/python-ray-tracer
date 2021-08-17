from PIL import Image
from datatypes import *
import time
import numpy as np
from numpy import dot
from numpy.linalg import norm
from numpy import sqrt, arctan2, arccos, sin, tan
from math import radians, degrees, inf
from random import uniform
import ray


FLOAT = 'float32'


def array(tuple):
	return np.array(tuple, dtype=FLOAT)


# testing
# while True:
#	x = Op.urandom()
#	if Op.dot(x,x) > 1: break
#	print(Op.dot(x,x))
# exit()
none = np.array((0, 0), dtype='int')
WHITE = Color(array((1, 1, 1)))
L_BLUE = Color(array((0.5, 0.7, 1.0)))
RED = Color(array((1, 0, 0)))
GREEN = Color(array((0, 1, 0)))
BLUE = Color(array((0, 0, 1)))
BLACK = Color(array((0, 0, 0)))




#performance
MSAA = True
number_of_samples = 2
DOWN = 2
depth = 2
lens_rad = 0
DARKNESS=25#max is 255 Dont set 0
CORES = 8 #Set to your max core count

#constants
minimum = 0.01
maximum = inf
UP = array((0, 1, 0))
ASPECT_RATIO = float(16/9)
IMAGE_WIDTH = int(1920/DOWN)
IMAGE_HEIGHT = int(IMAGE_WIDTH/ASPECT_RATIO)
MAP = np.zeros((IMAGE_HEIGHT, IMAGE_WIDTH, 3), dtype='uint8')

#Materials
groundmt = Material('Diffuse', Color(array((1  ,1   ,1   ))),0.25,10)
leftmt =   Material('Diffuse', Color(array((0.8  , 0.9, 0))))
rightmt =  Material('Diffuse', Color(array((0  , 0.4, 0.4))),10)
centremt = Material('Diffuse', Color(array((0, 0.2, 0.1))),1)
smallmt =  Material('Diffuse', Color(array((0.2, 0.7, 0.9))))
backmt =   Material('Diffuse', Color(array((0.1, 0.9, 0.2))))
lsmallmt = Material('Diffuse',Color(array((0  , 0.3  ,0.2 ))),0.5)
lightmt1 = Material('Light', Color(array((1, 1, 0))), 1)
lightmt2 = Material('Light', Color(array((0.5, 0  ,0.5 ))), 1)
#Objects(Circles as of now, ie:ver3.2)
centre = Circle(Point(array((0    , 0.25  , 0    ))), 0.75,centremt)
right =   Circle(Point(array((1   , 2     , -3   ))), 2.5, rightmt)
left =    Circle(Point(array((-5  , 0.325   , -10))), 1.2, leftmt)
back =    Circle(Point(array((3.5 , -0.1  , -3.25))), 0.5, backmt)
ground =  Circle(Point(array((0   , -200.5, 0    ))), 200, groundmt)
leftsmall=Circle(Point(array((-2  , 0     , 0    ))), 0.5, lsmallmt)
light=    Circle(Point(array((0   , 2.75  , -4      ))),1,  lightmt1)
smlight2= Circle(Point(array((0   , -0.5  , 1.5  ))),0.25,lightmt1)
smlight3= Circle(Point(array((-2  , -0.5  , 1    ))),      0.25,  lightmt1)
hittable = [ground, right, light, centre, leftsmall, back] #[centre,ground]

def Camera(VFOV, lookfrom, lookat, vup):
	f = lookfrom - lookat
	focus_dist = sqrt(f.dot(f))
	vup = Op.unit(vup)
	H = 1 * tan(radians(VFOV)/2)
	W = Op.unit(lookfrom - lookat)
	U = np.cross(vup, W)
	V = np.cross(W, U)
	VIEWPORT_HEIGHT = 4 * H
	VIEWPORT_WIDTH = int(VIEWPORT_HEIGHT * ASPECT_RATIO)
	FOCAL_LENGTH = sqrt(VIEWPORT_HEIGHT**2 + VIEWPORT_WIDTH**2)
	HORIZONTAL = focus_dist * VIEWPORT_WIDTH * U
	VERTICAL = focus_dist * VIEWPORT_HEIGHT * V
	LOWER_LEFT_CORNER = lookfrom - HORIZONTAL/2 - \
		VERTICAL/2 - focus_dist * W*FOCAL_LENGTH
	# print(HORIZONTAL,VERTICAL,LOWER_LEFT_CORNER)
	return HORIZONTAL, VERTICAL, LOWER_LEFT_CORNER, U, V



# MAIN


def set_face_normal(ray, outward_normal):
	dp = dot(ray.vec, outward_normal.vec)
	front_face = dp/abs(dp)  # -ve is True
	return  front_face, -front_face*outward_normal.vec

# WIP


def find_nearest(ray, Min, Max, hittable):
	hit_obj = None
	hit_distance = Max
	surface_info = 0
	for i in hittable:
		glassinfo, corrected_vector, t = hit_check(ray, i.centre, i.radius, Min, Max)
		if t > 0 and t < hit_distance:
			surface_info = corrected_vector
			hit_obj = i
			hit_distance = t
	return hit_obj, surface_info, hit_distance, glassinfo

# tried


def hit_check(ray, centre, radius, Min, Max):
	oc = ray.origin - centre
	a = dot(ray.dir, ray.dir)
	b = dot(oc, ray.dir)
	c = dot(oc, oc) - (radius*radius)
	discriminant = (b*b)-(a*c)
	if discriminant > 0:
		high = -b/a
		low = sqrt(discriminant)/a
		temp = high - low
		if temp < Max and temp > Min:
			pt = Op.at(ray, temp)
			n = Vector(Op.unit(pt - centre))
			x,y=set_face_normal(ray, n)
			return x,y,temp
		temp = high + low
		if temp < Max and temp > Min:
			pt = Op.at(ray, temp)
			n = Vector(Op.unit(pt - centre))
			x,y=set_face_normal(ray, n)
			return x,y,temp
		return 0, 0, 0
	else:
		return 0, 0, 0


def set_map(ray, max_depth, OBJs):
	if max_depth <= 0:
		return BLACK.col
	obj, surf_info, hit_distance, glassetachanger = find_nearest(ray, minimum, maximum, OBJs)
	if hit_distance is not inf and obj is not None : #type(surf_info) != int:
		hit_point = Op.at(ray, hit_distance)
		if obj.material.type == 'Metal':
			dot_bool, target = Metal(surf_info, ray, obj.material.info)
		elif obj.material.type == 'Diffuse':
			dot_bool, target = Diffuse(surf_info, ray, hit_point, obj.material.info)
		elif obj.material.type == 'Glass':
			dot_bool, target = Glass(surf_info,glassetachanger, ray, hit_point, obj.material.info)
		else:
			return obj.material.color.col*obj.material.info
		N = Vector(-Op.unit(hit_point - obj.centre))
		if dot_bool == True: #always True. Remove if later
			ray_color = WHITE.col # for glass WIP... but depreciated now
			if not obj.material.type == 'Glass':#WIP... gonna depreciate glass until i get it to work
				ray_color = obj.material.color.col
				if obj.material.texture != 0:
					theta = arctan2(N.x, N.z)
					phi = arccos(N.y)
					raw_u = theta / (2*pi)
					u = 1 - raw_u
					v = 1 - phi/pi
					if (int(u*20) + int(v*400)) % 2 == 0:
						ray_color = obj.material.color.col/2
					else:
						ray_color = obj.material.color.col
			return (ray_color)*set_map(Ray(hit_point, Vector(target-hit_point).vec), max_depth-1, OBJs)
		else:
			print(dot_bool)
			bgy = ray.diry
			bgx = ray.dirx

			return ((1-bgy)*(L_BLUE.col) + bgx*WHITE.col)
	else:
		bg = ray.diry
		return ((1-bg)*(L_BLUE.col) + bg*WHITE.col)/DARKNESS  # This is BG array((0  , 0  , 0    ))

def gamma_correct(color):
	return np.sqrt(color)

def Light():
	pass

def Diffuse(surf_info, ray, hit_pt, fuzz):
	return True, hit_pt + surf_info + Op.urandomacc()*fuzz


def Metal(surf_info, ray, fuzz):
	normal = surf_info
	reflected_ray = Op.reflect(ray.vec, normal)
	return True, (reflected_ray + fuzz*Op.urandom())


def Glass(surf_info, etachanger, ray, hit_pt, eta):
	#surf_info is currently a vector or int(0) change back to array if working on glass. Note:WIP... surfinfo changed to surfinfo(corrected ray) & etachanger
	n = -surf_info
	r = ray.dir
	e = eta**(-etachanger)
	'''
	ray_in = ray.dir
	e = eta**(-etachanger)
	cos_x = dot(-ray_in, n) #/(norm(ray_in))
	sin_x = sqrt(1 - cos_x**2)
	'''
	if dot(n,r)<0: #abs(e*sin_x) > 1:
		print("reflected")
		return True, Op.reflect(ray.vec, n)
	# elif uniform(0,1) > Op.schlick(cos_x, etai_by_etar):
		# return True, Op.reflect(ray.vec, n)
	else:
		return True, Op.refract(r, n, e)


def get_ray(O, LLC, u, H, v, V, cam_info):
	rd = Vector(cam_info * Op.unit(ray_in_unit_disk()))
	offset = 0  # (10**-shift)*(cam_info[1]*rd.x+ cam_info[2]*rd.y)
	return Ray(O + offset, LLC + u*H + v*V - O - offset)


def ray_in_unit_disk():
	return array((uniform(-1, 1), uniform(-1, 1), 0))

@ray.remote
def msaa(info, depth, objs):
	rows=np.zeros((info[3],3), dtype='uint8')
	for i in range(info[3]):
		info[1] = i
		color = array((0, 0, 0))
		for k in range(1, info[0]+1):
			u = ((info[1]+(k/info[0]))/(info[3]-1))
			v = ((info[2]+(k/info[0]))/(info[4]-1))
			r = get_ray(info[5], info[6], u, info[7], v, info[8], info[9])
			color += set_map(r, depth, objs)
		rows[i] = 255*gamma_correct(color/info[0])
	print(f'lines left:{info[2]}   {color[0]}     {int(100*info[10]/info[4])}%')
	return rows
		
# RENDER

def main(O, LLC, H, V, IH, IW, objs, cam_info, shift):
	msaainfo = [number_of_samples, 0, 0, IW, IH, O, LLC, H, V, cam_info, 0]
	FRAMELIST=[]
	for j in range(IH):
		y = IH - j - 1
		msaainfo[2] = y
		msaainfo[10] = j
		'''
		color=[]
		for i in range(IW):
			msaainfo[1] = i
			color.append(msaa.remote(msaainfo, depth, objs))
		color=np.array(ray.get(color))
		MAP[j] = 255*gamma_correct(color/number_of_samples)
		'''
		FRAMELIST.append(msaa.remote(msaainfo, depth, objs))
		

	FRAME=np.array(ray.get(FRAMELIST), dtype='uint8')
	#print('saving images')
	im = Image.fromarray(FRAME)
	# im.save(f"{shift}.png")
	finish = time.time()
	im = im.resize((round(im.size[0]*DOWN), round(im.size[1]*DOWN)))
	#img = "rt_test_0_{DOWN}_{number_of_samples}_{int(finish-start)}s_N.png"
	#im.show()
	#im.save(f"{i}.png")
	im.save(f"rt_test_0_{DOWN}_{number_of_samples}_{int(finish-start)}s_N.png")
	#im.show(f"rt_test_0_{DOWN}_{number_of_samples}_{int(finish-start)}s_N.png")#f"rt_test_0_{DOWN}_{number_of_samples}_{int(finish-start)}s_N.png"


if __name__ == '__main__':
	#print(ORIGIN.pt, LOWER_LEFT_CORNER.vec, HORIZONTAL.vec, VERTICAL.vec, IMAGE_HEIGHT, IMAGE_WIDTH, hittable)
	t = time.time()
	ray.init(num_cpus=CORES)
	print('ray_initialized in ', time.time()-t, 's')
	start = time.time()
	for i in range(1):
		print(i,"/360")
		O = array((7*sin(radians(i)), 2, 7*sin(radians(90-i))))
		H, V, LLC, u, v = Camera(120, O, array((0,0,0)), array((0,1,0)))
		main(O, LLC, H, V, IMAGE_HEIGHT, IMAGE_WIDTH,
			 hittable, lens_rad, i) #np.array((lens_rad, u, v))<- matters when offset needs to be set in get_ray. Goes in place of lens_rad
	print(f'{time.time()-start}s taken')
	exit()

	print('CREATING GIF')
	if True:
		images = []
		for i in range(360):
			images.append(imageio.imread(f'{i}.png'))
		# for i in range(9):
			#m = 8 - i
			# images.append(imageio.imread(f'{m}.png'))
		imageio.mimsave('test_RT_1.gif', images, duration=0.05)
	print('GIF CREATED')
