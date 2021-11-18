"""
Braking simulation environment
Based on a modified version of the car_racing environment (mostly just about the graphic).
The kinematics and the environment
State consists of 5 columns:
vehicle position y, vehicle velocity y, pedestrian posistion x, pedestrian position y, pedestrian velocity x.
(veh_pos_y, veh_vel_y, ped_pos_x, ped_pos_y, ped_vel_x)
The vehicle is moving along the y direction only (unless the vehicle is drifting or bumping an object),
while the pedestrian is only moving along the x axis (crossing the road) in current setting. The track is a fixed
straight track going along the y direction, and the vehical starts from (0, 0) in every episode.
Please note that, in the paper (Chae et al. 2017), the car is moving along the x direction,
so the definition of x, y axis is flipped in this module.
The episode finishes when the following conditions are reached:
1. The vehicle passes the pedestrian.
2. The vehicle bumps the pedestrian.
3. The vehicle is completely stopped.
Licensed on the same terms as the rest of OpenAI Gym.
"""
import sys
import math
import numpy as np
from random import randint, random

import Box2D
from Box2D.b2 import fixtureDef
from Box2D.b2 import polygonShape
from Box2D.b2 import contactListener

import gym
from gym import spaces
# costumed object classes modified from car_dynamics
from .dynamics.brake_dynamics import Senior_Car, Pedestrian, marker
from gym.utils import seeding, EzPickle

import pyglet

#pyglet.options["debug_gl"] = False
from pyglet import gl
import pdb

STATE_W = 96  # less than Atari 160x192
STATE_H = 96
VIDEO_W = 400
VIDEO_H = 600
WINDOW_W = 400
WINDOW_H = 600

SCALE = 6.0  # Track scale
TRACK_RAD = 1000 / SCALE  # Track is heavily morphed circle with this radius
PLAYFIELD = 2000 / SCALE  # Game over boundary
# Frames per second, following the setting in Chae et al 2017 (section IV, A, page 4)
FPS = 10
ZOOM = 1.0  # Camera zoom
ZOOM_FOLLOW = False  # Set to False for fixed view (don't use zoom)


#TRACK_DETAIL_STEP = 21 / SCALE
#TRACK_TURN_RATE = 0.31
TRACK_WIDTH = 5.0
#BORDER = 8 / SCALE
#BORDER_MIN_COUNT = 4


ROAD_COLOR = [0.4, 0.4, 0.4]

# ==== Chae et al. 2017 parameters =====

TTC_LOW = 1.5         # min time-to-collision (sec)
TTC_VAR = 2.5         # (max-min) time-to-collision (sec)
VEH_VEL_BASE = 10/3.6  # min velocity of vehicle
VEH_VEL_VAR = 50/3.6  # (max-min) velocity of vehicle
PED_VEL_BASE = 2  # min velocity of pedestrian
PED_VEL_VAR = 0.5  # (max - min) velocity of pedestrian

# ---- Reward function parameters (see eq. 8 in the paper) ----
ALPHA = 10.0**(-3.5)  # 0.001
BETA = 0.1
EPSILON = 0.01
LAMBDA = 100


class FrictionDetector(contactListener):
    def __init__(self, env):
        contactListener.__init__(self)
        self.env = env

    def BeginContact(self, contact):
        self._contact(contact, True)

    def EndContact(self, contact):
        self._contact(contact, False)

    def _contact(self, contact, begin):
        tile = None
        obj = None
        u1 = contact.fixtureA.body.userData
        u2 = contact.fixtureB.body.userData
        if u1 and "road_friction" in u1.__dict__:
            tile = u1
            obj = u2
        if u2 and "road_friction" in u2.__dict__:
            tile = u2
            obj = u1
        if not tile:
            return

        tile.color[0] = ROAD_COLOR[0]
        tile.color[1] = ROAD_COLOR[1]
        tile.color[2] = ROAD_COLOR[2]
        if not obj or "tiles" not in obj.__dict__:
            return
        if begin:
            obj.tiles.add(tile)
            if not tile.road_visited:
                tile.road_visited = True
                #self.env.reward += 1000.0 / len(self.env.track)
                self.env.tile_visited_count += 1
        else:
            obj.tiles.remove(tile)


class SCADASEnv(gym.Env, EzPickle):
    metadata = {
        "render.modes": ["human", "rgb_array", "state_pixels"],
        "video.frames_per_second": FPS,
    }

    def __init__(self, verbose=1):
        EzPickle.__init__(self)
        self.seed()
        self.contactListener_keepref = FrictionDetector(self)
        self.world = Box2D.b2World(
            (0, 0), contactListener=self.contactListener_keepref)
        self.viewer = None
        self.invisible_state_window = None
        self.invisible_video_window = None
        self.road = None
        self.car = None   # vehicle
        self.ped = None   # pedestrian
        self.brake_mag = [-9.8, -5.9, -2.9, 0]  # in m/s^2
        self.reward = 0.0
        self.decel_reward_acc = 0.0
        self.bump_reward_acc = 0.0
        self.stop_reward_acc = 0.0
        self.passing_reward_acc = 0.0
        self.prev_reward = 0.0
        self.ttc = 0.0    # Time-to-collision
        self.ped_pos_ini = 0.0
        self.verbose = verbose
        self.fd_tile = fixtureDef(
            shape=polygonShape(vertices=[(0, 0), (1, 0), (1, -1), (0, -1)])
        )

        # high (#5), mid, low brake and none (constant speed)
        self.action_space = spaces.Discrete(4)


        self.observation_space = spaces.Box(
            np.float32(np.array([0, 0, -1, 0, -1])), np.float32(np.array([1, 1, 1, 1, 1])))  # TOEXAM

        self.state = None   # veh_pos_y, veh_vel_y, ped_pos_x, ped_pos_y, ped_vel_x

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def _destroy(self):
        if not self.road:
            return
        for t in self.road:
            self.world.DestroyBody(t)
        self.road = []
        self.car.destroy()
        self.ped.destroy()

    def _create_track(self):

        # Create a straight track at x = 0 across the image frame along the y direction
        self.road = []
        track = []
        for i in range(100):
            track.append([0, 0, 0, -50*TRACK_WIDTH+i*TRACK_WIDTH])
        # Create tiles
        for i in range(len(track)):
            alpha1, beta1, x1, y1 = track[i]
            alpha2, beta2, x2, y2 = track[i - 1]
            road1_l = (
                x1 - TRACK_WIDTH * math.cos(beta1),
                y1 - TRACK_WIDTH * math.sin(beta1),
            )
            road1_r = (
                x1 + TRACK_WIDTH * math.cos(beta1),
                y1 + TRACK_WIDTH * math.sin(beta1),
            )
            road2_l = (
                x2 - TRACK_WIDTH * math.cos(beta2),
                y2 - TRACK_WIDTH * math.sin(beta2),
            )
            road2_r = (
                x2 + TRACK_WIDTH * math.cos(beta2),
                y2 + TRACK_WIDTH * math.sin(beta2),
            )
            vertices = [road1_l, road1_r, road2_r, road2_l]
            self.fd_tile.shape.vertices = vertices
            t = self.world.CreateStaticBody(fixtures=self.fd_tile)
            t.userData = t
            c = 0.01 * (i % 3)
            t.color = [ROAD_COLOR[0] + c, ROAD_COLOR[1] + c, ROAD_COLOR[2] + c]
            t.road_visited = False
            t.road_friction = 1.0
            t.fixtures[0].sensor = True
            self.road_poly.append(
                ([road1_l, road1_r, road2_r, road2_l], t.color))
            self.road.append(t)
        self.track = track
        return True

    def reset(self):
        self._destroy()
        self.reward = 0.0
        self.decel_reward_acc = 0.0
        self.bump_reward_acc = 0.0
        self.stop_reward_acc = 0.0
        self.passing_reward_acc = 0.0        
        self.prev_reward = 0.0
        self.tile_visited_count = 0
        self.t = 0.0
        self.road_poly = []

        while True:
            success = self._create_track()
            if success:
                break
            if self.verbose == 1:
                print(
                    "retry to generate track (normal if there are not many"
                    "instances of this message)"
                )

        # === Setting up the initial state following Chae et al 2017 (section IV, A, page 4-5)

        veh_vel = VEH_VEL_BASE + random() * VEH_VEL_VAR  # Initial velocity of the vehicle

        # Pedestrian crossing scenario selection
        # In the paper, the original setting is that for 1,2 = cross / 3,4 = stay
        # However, here I change it to 1,3 = cross from the left side of the road / 2,4 = cross from the right side of the road
        scenario_idx = randint(1, 4)
        ped_vel = (PED_VEL_BASE + PED_VEL_VAR * random()) * \
            (-1.0)**(scenario_idx+1)  # Initial velocity of the pedestrian

        # Initial position of pedestrian
        # Initial position of the pedestrian
        ped_pos = [TRACK_WIDTH*(-1.0)**scenario_idx, veh_vel*5]
        self.ped_pos_ini = veh_vel*5
        ttc = TTC_LOW + random() * TTC_VAR  # TTC for the episode
        ped_trig = (5.0 - ttc) * veh_vel  # Ped. trigger point

        self.ttc = ttc
        # Create a car object using the Senior_Car class in "adas_dynamics.py"
        self.car = Senior_Car(self.world, 0.0, 0, 0, 0, veh_vel, 0.0)
        # Create a pedestrian object using the Pedestrian class in "adas_dynamics.py"
        self.ped = Pedestrian(self.world, 3.1415/2.0*(-1.0)
                              ** scenario_idx, ped_pos[0], ped_pos[1], 0.0, 0.0)

        # The pedestrian only start to cross the road when the vehicle reach the trigger point (trigger_dist)
        self.ped.trig_vel = 5.0/ttc * (-1.0)**(scenario_idx+1)  # ped_vel
        self.ped.trig_dist = (5.0 - ttc) * veh_vel
        self.ped.scenario_idx = scenario_idx
        self.ped.pos_x_ini = TRACK_WIDTH*(-1.0)**scenario_idx

        # This is used for marking the trigger point, but it can be removed or changed into a painted mark instead of a real object
        self.mak = marker(self.world, 3.1415/2.0*(-1.0) **
                          scenario_idx, TRACK_WIDTH/1.3, self.ped.trig_dist)

        # The state of the simulation
        self.state = [self.car.hull.position[1]/((VEH_VEL_BASE+VEH_VEL_VAR)*5.0), self.car.hull.linearVelocity[1]/((VEH_VEL_BASE+VEH_VEL_VAR)),
                      self.ped.hull.position[0]/(TRACK_WIDTH), self.ped.hull.position[1]/((VEH_VEL_BASE+VEH_VEL_VAR)*5.0), self.ped.hull.linearVelocity[0]/(PED_VEL_BASE + PED_VEL_VAR)]  # veh_pos_y, veh_vel_y, ped_pos_x, ped_pos_y, ped_vel_x
        # pdb.set_trace()

        return np.array(self.state)

    def step(self, action):

        veh_pos_y, veh_vel_y, ped_pos_x, ped_pos_y, ped_vel_x = self.state
        veh_pos_y = veh_pos_y * ((VEH_VEL_BASE+VEH_VEL_VAR)*5.0)
        veh_vel_y = veh_vel_y * ((VEH_VEL_BASE+VEH_VEL_VAR))
        ped_pos_x = ped_pos_x * (TRACK_WIDTH)
        ped_pos_y = ped_pos_y * ((VEH_VEL_BASE+VEH_VEL_VAR)*5.0)
        ped_vel_x = ped_vel_x * (PED_VEL_BASE + PED_VEL_VAR)

        # Distance between the vehicle and the pedestrian
        dist = np.sqrt((veh_pos_y - ped_pos_y) ** 2.0 + ped_pos_x ** 2.0)

        if self.car.hull.position[1] >= self.ped.trig_dist:
            self.ped.trigger = True
        if(abs(self.ped.hull.position[0]) > (TRACK_WIDTH*1.1)):
            self.ped.trigger = False

        # The deceleration applied based on the action taken.
        #decel = self.brake_mag[action]
        decel = self.brake_mag[action]
        self.car.acc = decel

        # Update for each time step
        self.car.step(1.0 / FPS)
        self.ped.step(1.0 / FPS, self.car.hull.position[1], min(
            abs(self.ped.pos_x_ini - ped_pos_x) / (TRACK_WIDTH*2.1), 1))
        self.world.Step(1.0 / FPS, 6 * 30, 2 * 30)
        self.t += 1.0 / FPS

        # Store the updated state
        self.state = [self.car.hull.position[1]/((VEH_VEL_BASE+VEH_VEL_VAR)*5.0), self.car.hull.linearVelocity[1]/((VEH_VEL_BASE+VEH_VEL_VAR)),
                      self.ped.hull.position[0]/(TRACK_WIDTH), self.ped.hull.position[1]/((VEH_VEL_BASE+VEH_VEL_VAR)*5.0), self.ped.hull.linearVelocity[0]/(PED_VEL_BASE + PED_VEL_VAR)]  # veh_pos_y, veh_vel_y, ped_pos_x, ped_pos_y, ped_vel_x

        step_reward = 0
        done = False
        done_con = 0  # recording under which condition the simulation is terminated
        if action is not None:  # First step without action, called from reset()

            dist = np.sqrt((self.car.hull.position[0] - self.ped.hull.position[0]) ** 2.0 +
                           (self.car.hull.position[1] - self.ped.hull.position[1]) ** 2.0)   # distance between the vehicle and the pedestrian
            # velocity difference between time step t and t-1
            decel_v = veh_vel_y - self.car.hull.linearVelocity[1]
            cosineangle = (
                self.ped.hull.position[1] - self.car.hull.position[1])/dist

            # Reward function (see eq. 8 in the paper). Note that the minus sign of BETA is different than the original equation.
            veh_vel = np.sqrt(
                self.car.hull.linearVelocity[1] ** 2.0 + self.car.hull.linearVelocity[0] ** 2.0)
            alph = 0.01
            bet = 0
            safedist = 2
            decel_reward = -1.0* decel_v * (dist **3.0) * alph*0.2
            passing_reward = 1.5 * math.exp(-cosineangle)
            stop_reward = -1.0* decel_v * (- 1.0* (self.car.vel_ini**2.0/(2.0*5.9))**3.5/(dist/(6.0)+0.001)**1.0 ) * alph
            bump_reward = 0.0

            self.reward += decel_reward +stop_reward+passing_reward
            self.decel_reward_acc += decel_reward
            self.stop_reward_acc += stop_reward
            self.passing_reward_acc += passing_reward
            step_reward = self.reward - self.prev_reward
            self.prev_reward = self.reward

            # Check if the termination condition is met
            if dist <= 1.0 and cosineangle > 0.8:
                done = True
                # print('Bumping.....')
                self.reward += -10000
                self.bump_reward_acc += -10000
                done_con = 1            
            if self.car.hull.position[1] >= self.ped_pos_ini:
                done = True
                #print('Vehicle passing pedestrian.....')
                done_con = 0
            elif (self.car.hull.linearVelocity[1] <= 0.0):
                done = True
                #print('Vehicle stopped.....')
                done_con = 2

        # Storing additional info to be passed as outputs
        info = {'dist': dist, 'decel': decel_v, 'veh_vel': veh_vel, 'hull_vel': self.car.hull.linearVelocity, 'end_condition': done_con, 'ttc': self.ttc, 'trigger_dist': self.ped.trig_dist,
                'ped_vel': self.ped.hull.linearVelocity[0], 'scenario_idx': self.ped.scenario_idx, 'ped_trigger': self.ped.trigger, 'decel_reward': self.decel_reward_acc, 'bump_reward': self.bump_reward_acc,
                'stop_reward': self.stop_reward_acc, 'passing_reward': self.passing_reward_acc}
        return np.array(self.state), self.reward, done, info

    def render(self, mode="human"):
        assert mode in ["human", "state_pixels", "rgb_array"]
        if self.viewer is None:
            from gym.envs.classic_control import rendering

            self.viewer = rendering.Viewer(WINDOW_W, WINDOW_H)
            self.score_label = pyglet.text.Label(
                "0000",
                font_size=36,
                x=20,
                y=WINDOW_H * 2.5 / 40.00,
                anchor_x="left",
                anchor_y="center",
                color=(255, 255, 255, 255),
            )
            self.transform = rendering.Transform()

        if "t" not in self.__dict__:
            return  # reset() not called yet

        # Animate zoom first second:
        zoom = ZOOM * SCALE
        vel = self.car.hull.linearVelocity
        self.transform.set_scale(zoom, zoom)
        self.transform.set_translation(WINDOW_W / 2, WINDOW_H / 20)

        # Draw objects
        self.car.draw(self.viewer, mode != "state_pixels")
        self.ped.draw(self.viewer, mode != "state_pixels")
        self.mak.draw(self.viewer, mode != "state_pixels")

        arr = None
        win = self.viewer.window
        win.switch_to()
        win.dispatch_events()

        win.clear()
        t = self.transform
        if mode == "rgb_array":
            VP_W = VIDEO_W
            VP_H = VIDEO_H
        elif mode == "state_pixels":
            VP_W = STATE_W
            VP_H = STATE_H
        else:
            pixel_scale = 1
            if hasattr(win.context, "_nscontext"):
                pixel_scale = (
                    win.context._nscontext.view().backingScaleFactor()
                )  # pylint: disable=protected-access
            VP_W = int(pixel_scale * WINDOW_W)
            VP_H = int(pixel_scale * WINDOW_H)

        gl.glViewport(0, 0, VP_W, VP_H)
        t.enable()
        self.render_road()
        for geom in self.viewer.onetime_geoms:
            geom.render()
        self.viewer.onetime_geoms = []
        t.disable()
        #self.render_indicators(WINDOW_W, WINDOW_H)

        if mode == "human":
            win.flip()
            return self.viewer.isopen

        image_data = (
            pyglet.image.get_buffer_manager().get_color_buffer().get_image_data()
        )
        arr = np.fromstring(image_data.get_data(), dtype=np.uint8, sep="")
        arr = arr.reshape(VP_H, VP_W, 4)
        arr = arr[::-1, :, 0:3]

        return arr

    def close(self):
        if self.viewer is not None:
            self.viewer.close()
            self.viewer = None

    def render_road(self):
        colors = [0.4, 0.8, 0.4, 1.0] * 4
        polygons_ = [
            +PLAYFIELD,
            +PLAYFIELD,
            0,
            +PLAYFIELD,
            -PLAYFIELD,
            0,
            -PLAYFIELD,
            -PLAYFIELD,
            0,
            -PLAYFIELD,
            +PLAYFIELD,
            0,
        ]

        k = PLAYFIELD / 20.0
        colors.extend([0.4, 0.9, 0.4, 1.0] * 4 * 20 * 20)
        for x in range(-20, 20, 2):
            for y in range(-20, 20, 2):
                polygons_.extend(
                    [
                        k * x + k,
                        k * y + 0,
                        0,
                        k * x + 0,
                        k * y + 0,
                        0,
                        k * x + 0,
                        k * y + k,
                        0,
                        k * x + k,
                        k * y + k,
                        0,
                    ]
                )

        for poly, color in self.road_poly:
            colors.extend([color[0], color[1], color[2], 1] * len(poly))
            for p in poly:
                polygons_.extend([p[0], p[1], 0])

        vl = pyglet.graphics.vertex_list(
            len(polygons_) // 3, ("v3f", polygons_), ("c4f", colors)  # gl.GL_QUADS,
        )
        vl.draw(gl.GL_QUADS)

    def render_indicators(self, W, H):
        s = W / 40.0
        h = H / 40.0
        colors = [0, 0, 0, 1] * 4
        polygons = [W, 0, 0, W, 5 * h, 0, 0, 5 * h, 0, 0, 0, 0]

        def vertical_ind(place, val, color):
            colors.extend([color[0], color[1], color[2], 1] * 4)
            polygons.extend(
                [
                    place * s,
                    h + h * val,
                    0,
                    (place + 1) * s,
                    h + h * val,
                    0,
                    (place + 1) * s,
                    h,
                    0,
                    (place + 0) * s,
                    h,
                    0,
                ]
            )

        def horiz_ind(place, val, color):
            colors.extend([color[0], color[1], color[2], 1] * 4)
            polygons.extend(
                [
                    (place + 0) * s,
                    4 * h,
                    0,
                    (place + val) * s,
                    4 * h,
                    0,
                    (place + val) * s,
                    2 * h,
                    0,
                    (place + 0) * s,
                    2 * h,
                    0,
                ]
            )

        true_speed = np.sqrt(
            np.square(self.car.hull.linearVelocity[0])
            + np.square(self.car.hull.linearVelocity[1])
        )

        vl = pyglet.graphics.vertex_list(
            len(polygons) // 3, ("v3f", polygons), ("c4f", colors)  # gl.GL_QUADS,
        )
        vl.draw(gl.GL_QUADS)
        self.score_label.text = "%04i" % self.reward
        self.score_label.draw()


if __name__ == "__main__":
    # pdb.set_trace()
    import time
    from pyglet.window import key

    a = np.array([0.0, 0.0, 0.0, 1.0])

    def key_press(k, mod):
        global restart
        if k == 0xff0d:
            restart = True
        if k == key.LEFT:
            a[0] = 1.0
            a[3] = 0.0
        if k == key.RIGHT:
            a[1] = 1.0
            a[3] = 0.0
        if k == key.UP:
            a[2] = 1.0
            a[3] = 0.0

    def key_release(k, mod):
        if k == key.LEFT:
            a[0] = 0
            a[3] = 1.0
        if k == key.RIGHT:
            a[1] = 0
            a[3] = 1.0
        if k == key.UP:
            a[2] = 0
            a[3] = 1.0

    ec_count = np.zeros(3)
    env = SCADASEnv()

    env.render()

    env.viewer.window.on_key_press = key_press
    env.viewer.window.on_key_release = key_release
    record_video = False
    if record_video:
        from gym.wrappers.monitor import Monitor

        env = Monitor(env, "/tmp/video-test", force=True)
    isopen = True
    while isopen:
        env.reset()
        total_reward = 0.0
        steps = 0
        restart = False
        # pdb.set_trace()
        r_p = 0
        while True:
            at = np.where(a == 1)[0][0]
            s, r, done, info = env.step(at)
            # the minimun distance you should keep stepping on the hard brake (-9.8 km/s) to avoid collision
            dtc_1 = info['veh_vel']**2.0/(2.0 * 9.8)
            # the minimun distance you should keep stepping on the mid brake (-5.9 km/s) to avoid collision
            dtc_2 = info['veh_vel']**2.0/(2.0 * 5.9)
            # the minimun distance you should keep stepping on the soft brake (-2.9 km/s) to avoid collision
            dtc_3 = info['veh_vel']**2.0/(2.0 * 2.9)

            #       info = {'dist':dist, 'decel':decel_v,'veh_vel':veh_vel,'hull_vel':car.hull.linearVelocity,'end_condition':done_con, 'ttc':self.ttc, 'trigger_dist':self.ped.trig_dist,
            # 'ped_vel':self.ped.hull.linearVelocity[0], 'scenario_idx':self.ped.scenario_idx,'ped_trigger':self.ped.trigger}

            r_p = r
            total_reward += r

            steps += 1
            isopen = env.render()
           # time.sleep(0.2)
            if done:
                #    print('#####DONE')
                ec_count[info['end_condition']] += 1
            #    print('End condition:', ec_count/float(np.sum(ec_count)))
            #    input("Press Enter to continue...")
            if done or restart or isopen == False:
                break
                # time.sleep(3.0)
    env.close()
