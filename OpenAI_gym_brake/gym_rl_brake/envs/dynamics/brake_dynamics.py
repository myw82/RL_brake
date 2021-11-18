"""
Braking dynamics simulation.

It is based a modified version of the "car_dynamics" but the kinematics is mostly just constant velocity and acceleration without torque.

"""

import numpy as np
import math
import Box2D
from Box2D.b2 import (edgeShape, circleShape, fixtureDef, polygonShape, revoluteJointDef, contactListener, shape)

SIZE = 0.010
WHEEL_R = 27
WHEEL_W = 14
WHEELPOS = [
    (-55, +80), (+55, +80),
    (-55, -82), (+55, -82)
    ]
HULL_POLY1 = [
    (-60, +130), (+60, +130),
    (+60, +110), (-60, +110)
    ]
HULL_POLY2 = [
    (-15, +120), (+15, +120),
    (+20, +20), (-20, 20)
    ]
HULL_POLY3 = [
    (+25, +20),
    (+50, -10),
    (+50, -40),
    (+20, -90),
    (-20, -90),
    (-50, -40),
    (-50, -10),
    (-25, +20)
    ]
HULL_POLY4 = [
    (-50, -120), (+50, -120),
    (+50, -90),  (-50, -90)
    ]
WHEEL_COLOR = (0.0,  0.0, 0.0)
WHEEL_WHITE = (0.3, 0.3, 0.3)
MUD_COLOR = (0.4, 0.4, 0.0)


class marker:
	"""
  	It is basically an circular object to mark the trigger point, but ideally it should not be an objects. 
  	Otherwise occasionally the vehicle may bump into this marker if it drifts.

	"""
	def __init__(self, world, init_angle, init_x, init_y):
		self.x = 0.0
		self.y = 0.0
		self.world = world
		self.hull = self.world.CreateDynamicBody(
			position=(init_x, init_y),
			angle=init_angle,
			fixtures=fixtureDef(
			shape=circleShape(pos=(0, 0), radius=0.4)),
			linearVelocity = (0.0, 0.0)
			)
		self.hull.color = (0.0, 0.8, 0.0)
		self.drawlist = [self.hull]
	
	def draw(self, viewer, draw_particles=True):
		from gym.envs.classic_control import rendering
		for obj in self.drawlist:
			for f in obj.fixtures:
				trans = f.body.transform
				if type(f.shape) is circleShape:
					t = rendering.Transform(translation=trans*f.shape.pos)
					viewer.draw_circle(f.shape.radius, 30, color=obj.color).add_attr(t)
					
	def destroy(self):
		self.world.DestroyBody(self.hull)
		self.hull = None


class Pedestrian:
	"""
   	A circular object representing a constant speed moving pedestrian crossing the road.
 
   	It should only start crossing when the vehicle passes the trigger point.
   	However, due to some unknown reason, I can't really just change the velocity after reaching the trigger point 
 	and it will start to move. Instead, I slowly increase the pedestrian speed so that it reached the desired speed 
 	when the vehicle reaches the trigger point.
 
	"""
	
	def __init__(self, world, init_angle, init_x, init_y, trig_vx, trig_vy):
		self.world = world
		self.hull = self.world.CreateDynamicBody(
			position=(init_x, init_y),
			angle=init_angle,
			fixtures=fixtureDef(
			shape=circleShape(pos=(0, 0), radius=0.6)),
			linearVelocity = (0.0, 0.0)
			)
		self.hull.color = (0.0, 0.3, 0.7)
		self.drawlist = [self.hull]
		self.trigger = False
		self.trig_vel = trig_vx
		self.trig_dist = 0.0
		self.pos_x_ini = 0.0		
		self.scenario_idx = 0
		
	def step(self, dt, dist, dist_x):		
		# It gradually increase the speed until the vehicle reaches the trigger point (full speed)	
		# Then it should stop when it crosses the road.
		if (dist / self.trig_dist)**3.0 < 0.003:
			pow = 2.8
			vel_diff = 0.70
		else:
			pow = 3.0
			vel_diff = 0.0
		self.hull.linearVelocity[0] += ((self.trig_vel - vel_diff - self.hull.linearVelocity[0]) * min(1 ,dist / self.trig_dist)**pow
										- (self.trig_vel - vel_diff - self.hull.linearVelocity[0]) * min(1 ,dist / self.trig_dist)**pow * dist_x
										- self.hull.linearVelocity[0] * dist_x)
		self.hull.linearVelocity[1] = 0.0

		
	def draw(self, viewer, draw_particles=True):
		from gym.envs.classic_control import rendering
		for obj in self.drawlist:
			for f in obj.fixtures:
				trans = f.body.transform
				t = rendering.Transform(translation=trans*f.shape.pos)
				viewer.draw_circle(f.shape.radius, 25, color=obj.color).add_attr(t)
	
	def destroy(self):
		self.world.DestroyBody(self.hull)
		self.hull = None


class Senior_Car:
	"""
	An object sympolizes a moving vehicle (senior car). 
   
	The motion includes constant speed and deceleration.
 
	"""
	
	def __init__(self, world, init_angle, init_x, init_y, init_vx, init_vy, acc):
		self.world = world
		self.acc = acc
		self.vel = init_vy
		self.vel_ini = init_vy
		self.hull = self.world.CreateDynamicBody(
			position=(init_x, init_y),
			angle=init_angle,
			fixtures=[
				fixtureDef(shape=polygonShape(vertices=[(x*SIZE, y*SIZE) for x, y in HULL_POLY1]), density=1.0),
				fixtureDef(shape=polygonShape(vertices=[(x*SIZE, y*SIZE) for x, y in HULL_POLY2]), density=1.0),
				fixtureDef(shape=polygonShape(vertices=[(x*SIZE, y*SIZE) for x, y in HULL_POLY3]), density=1.0),
				fixtureDef(shape=polygonShape(vertices=[(x*SIZE, y*SIZE) for x, y in HULL_POLY4]), density=1.0)],
			linearVelocity = (init_vx, init_vy)
			)
		self.hull.color = (0.8, 0.0, 0.0)
		self.wheels = []
		WHEEL_POLY = [
			(-WHEEL_W, +WHEEL_R), (+WHEEL_W, +WHEEL_R),
			(+WHEEL_W, -WHEEL_R), (-WHEEL_W, -WHEEL_R)]
		
		for wx, wy in WHEELPOS:
			front_k = 1.0 if wy > 0 else 1.0
			w = self.world.CreateDynamicBody(
				position=(init_x+wx*SIZE, init_y+wy*SIZE),
				angle=init_angle,
				fixtures=fixtureDef(
					shape=polygonShape(vertices=[(x*front_k*SIZE,y*front_k*SIZE) for x, y in WHEEL_POLY]),
					density=0.1),
				linearVelocity = (init_vx, init_vy)
				)
			
			w.wheel_rad = front_k*WHEEL_R*SIZE
			w.color = WHEEL_COLOR
			#rjd = revoluteJointDef(
			#	bodyA=self.hull,
			#	bodyB=w,
			#	localAnchorA=(wx*SIZE, wy*SIZE),
			#	localAnchorB=(0,0),
			#	enableMotor=False,
			#	enableLimit=False,
			#	maxMotorTorque=0,
			#	motorSpeed=0,
			#	lowerAngle=0.0,
			#	upperAngle=0.0,
			#	)
			#w.joint = self.world.CreateJoint(rjd)
			#w.tiles = set()
			#w.userData = w
			self.wheels.append(w)
		self.drawlist = self.wheels + [self.hull] #self.wheels + 
	
	def step(self, dt):
		# Assuming the acceleration is always along the y direction
		# Updating the speed by applying deceleration
		self.hull.linearVelocity[1] += np.float32(self.acc * dt)
		self.hull.linearVelocity[0] = 0.0
		self.hull.angle = 0.0
		for w in self.wheels:
			w.linearVelocity[1] = self.hull.linearVelocity[1]#+= np.float32(self.acc * dt)
			w.linearVelocity[0] = 0.0
			w.angle = 0.0
	
	def draw(self, viewer, draw_particles=True):
		for obj in self.drawlist:
			for f in obj.fixtures:
				trans = f.body.transform
				path = [trans*v for v in f.shape.vertices]
				viewer.draw_polygon(path, color=obj.color)
				if "phase" not in obj.__dict__: continue
				a1 = 0.0#obj.phase
				a2 = 0.0+1.2#obj.phase + 1.2  # radians
				s1 = math.sin(a1)
				s2 = math.sin(a2)
				c1 = math.cos(a1)
				c2 = math.cos(a2)
				if s1 > 0 and s2 > 0: continue
				if s1 > 0: c1 = np.sign(c1)
				if s2 > 0: c2 = np.sign(c2)
				white_poly = [
					(-WHEEL_W*SIZE, +WHEEL_R*c1*SIZE), (+WHEEL_W*SIZE, +WHEEL_R*c1*SIZE),
					(+WHEEL_W*SIZE, +WHEEL_R*c2*SIZE), (-WHEEL_W*SIZE, +WHEEL_R*c2*SIZE)
					]
				#viewer.draw_polygon([trans*v for v in white_poly], color=WHEEL_WHITE)
	
	def destroy(self):
		self.world.DestroyBody(self.hull)
		self.hull = None
		for w in self.wheels:
			self.world.DestroyBody(w)
		self.wheels = []

