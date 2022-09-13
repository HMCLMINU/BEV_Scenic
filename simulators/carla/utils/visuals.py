#!/usr/bin/env python3

# This file utilizes existing CARLA code:

# Copyright (c) 2018 Intel Labs.
# authors: German Ros (german.ros@intel.com)
#
# This work is licensed under the terms of the MIT license.
# Permission is hereby granted, free of charge, to any person
# obtaining a copy of this software and associated documentation
# files (the "Software"), to deal in the Software without
# restriction, including without limitation the rights to use,
# copy, modify, merge, publish, distribute, sublicense, and/or
# sell copies of the Software, and to permit persons to whom the
# Software is furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be
# included in all copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
# EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
# MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
# NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS
# BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN
# ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN
# CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

import datetime
import math
import weakref
import pygame
import numpy as np
import carla
from carla import ColorConverter as cc
import cv2 as cv
import os,sys
import collections
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))
from carla_birdeye_view.__init__ import (
    BirdViewProducer,
    BirdView,
    DEFAULT_HEIGHT,
    DEFAULT_WIDTH,
    BirdViewCropType,
)
from carla_birdeye_view.mask import PixelDimensions

image_save_path ='/home/hmcl/Vehicle_Collision_Prediction_Using_convLSTM/for_project_data/'
image_save_path_BEV = '/home/hmcl/carla-birdeye-view/bev_datagen/carla-birdeye-view/autodata/'

IM_H, IM_W = (420, 280)
col_intensity = 0
seq_len = 15 # 3sec & 5hz
collision_folder = 0
safe_folder = 0 
def get_actor_display_name(actor, truncate=250):
    name = ' '.join(actor.type_id.replace('_', '.').title().split('.')[1:])
    return (name[:truncate-1] + u'\u2026') if len(name) > truncate else name

def save(bgr_img, img_count, n_seq, label, ego, folder):
	accel = math.sqrt(ego.get_acceleration().x**2 + ego.get_acceleration().y**2) 
	vel = math.sqrt(ego.get_velocity().x**2 + ego.get_velocity().y**2)
	yaw_rate = ego.get_angular_velocity().z
	if folder == 'camera':
		gray = cv.cvtColor(bgr_img, cv.COLOR_BGR2GRAY)
		# if collision, then save whole history
		np.savez("{}".format(os.path.join(image_save_path + str(n_seq) + '/{}'.format(img_count))), np.array(gray), [label], [accel, vel, yaw_rate])
		# cv.imwrite(os.path.join(image_save_path, str(n_seq),'{}.png'.format(img_count)), gray)
	elif folder == "bev":
		np.savez("{}".format(os.path.join(image_save_path_BEV + str(n_seq) + '/{}'.format(img_count))), np.array(bgr_img), [label], [accel, vel, yaw_rate])
		# cv.imwrite(os.path.join(image_save_path_BEV, str(n_seq),'{}.png'.format(img_count)), bgr_img)


# ==============================================================================
# -- HUD -----------------------------------------------------------------------
# ==============================================================================

class HUD(object):
	def __init__(self, width, height):
		self.dim = (width, height)
		font = pygame.font.Font(pygame.font.get_default_font(), 20)
		fonts = [x for x in pygame.font.get_fonts() if 'mono' in x]
		default_font = 'ubuntumono'
		mono = default_font if default_font in fonts else fonts[0]
		mono = pygame.font.match_font(mono)
		self._font_mono = pygame.font.Font(mono, 14)
		self._notifications = FadingText(font, (width, 40), (0, height - 40))
		self.server_fps = 0
		self.frame = 0
		self.simulation_time = 0
		self._info_text = []
		self._server_clock = pygame.time.Clock()

	def on_world_tick(self, timestamp):
		self._server_clock.tick()
		self.server_fps = self._server_clock.get_fps()
		self.frame = timestamp.frame
		self.simulation_time = timestamp.elapsed_seconds

	def tick(self, world, ego):
		if ego.carlaActor is None:
			return  # ego not spawned yet

		t = ego.carlaActor.get_transform()
		v = ego.carlaActor.get_velocity()
		c = ego.carlaActor.get_control()

		heading = 'N' if abs(t.rotation.yaw) < 89.5 else ''
		heading += 'S' if abs(t.rotation.yaw) > 90.5 else ''
		heading += 'E' if 179.5 > t.rotation.yaw > 0.5 else ''
		heading += 'W' if -0.5 > t.rotation.yaw > -179.5 else ''

		#colhist = collisionSensor.get_collision_history()
		#collision = [colhist[x + self.frame - 200] for x in range(0, 200)]
		#max_col = max(1.0, max(collision))
		#collision = [x / max_col for x in collision]

		vehicles = world.get_actors().filter('vehicle.*')
		pedestrians = world.get_actors().filter('walker.pedestrian.*')

		self._info_text = [
			'Server:  % 16d FPS' % self.server_fps,
			'Map:	 % 20s' % world.get_map().name,
			'Simulation time: % 12s' % datetime.timedelta(seconds=int(self.simulation_time)),
			'',
			'Number of vehicles: % 8d' % len(vehicles),
			'Number of pedestrians: % 8d' % len(pedestrians),
			'',
			'Ego: % 20s' % get_actor_display_name(ego.carlaActor, truncate=20),
			'',
			'Speed:   % 15.0f m/s' % math.sqrt(v.x**2 + v.y**2 + v.z**2),
			u'Heading:% 16.0f\N{DEGREE SIGN} % 2s' % (t.rotation.yaw, heading),
			'Location:% 20s' % ('(% 5.3f, % 5.3f)' % (t.location.x, t.location.y)),
			'Height:  % 18.0f m' % t.location.z,
		]

		try:
			_control_text = [
				'',
				('Throttle:', c.throttle, 0.0, 1.0),
				('Steer:', c.steer, -1.0, 1.0),
				('Brake:', c.brake, 0.0, 1.0),
				('Reverse:', c.reverse),
				('Hand brake:', c.hand_brake),
				('Manual:', c.manual_gear_shift),
				'Gear:		%s' % {-1: 'R', 0: 'N'}.get(c.gear, c.gear),
				#'',
				#'Collision:', collision,
			]
		except:
			_control_text = []
		finally:
			self._info_text.extend(_control_text)

		if len(vehicles) > 1:
			self._info_text += ['Nearby vehicles:']
			distance = lambda l: math.sqrt((l.x - t.location.x)**2 + (l.y - t.location.y)**2 + (l.z - t.location.z)**2)
			vehicles = [(distance(x.get_location()), x)
						for x in vehicles if x.id != ego.carlaActor.id]
			for d, vehicle in sorted(vehicles):
				if d > 200.0:
					break
				vehicle_type = get_actor_display_name(vehicle, truncate=22)
				self._info_text.append('% 4dm %s' % (d, vehicle_type))

	def render(self, display):
		info_surface = pygame.Surface((220, self.dim[1]))
		info_surface.set_alpha(100)
		display.blit(info_surface, (0, 0))
		v_offset = 4
		bar_h_offset = 100
		bar_width = 106
		for item in self._info_text:
			if v_offset + 18 > self.dim[1]:
				break
			if isinstance(item, list):
				if len(item) > 1:
					points = [(x + 8, v_offset + 8 + (1.0 - y) * 30) for x, y in enumerate(item)]
					pygame.draw.lines(display, (255, 136, 0), False, points, 2)
				item = None
				v_offset += 18
			elif isinstance(item, tuple):
				if isinstance(item[1], bool):
					rect = pygame.Rect((bar_h_offset, v_offset + 8), (6, 6))
					pygame.draw.rect(display, (255, 255, 255), rect, 0 if item[1] else 1)
				else:
					rect_border = pygame.Rect((bar_h_offset, v_offset + 8), (bar_width, 6))
					pygame.draw.rect(display, (255, 255, 255), rect_border, 1)
					f = (item[1] - item[2]) / (item[3] - item[2])
					if item[2] < 0.0:
						rect = pygame.Rect((bar_h_offset + f * (bar_width - 6), v_offset + 8), (6, 6))
					else:
						rect = pygame.Rect((bar_h_offset, v_offset + 8), (f * bar_width, 6))
					pygame.draw.rect(display, (255, 255, 255), rect)
				item = item[0]
			if item: # At this point has to be a str
				surface = self._font_mono.render(item, True, (255, 255, 255))
				display.blit(surface, (8, v_offset))
			v_offset += 18


# ==============================================================================
# -- FadingText ----------------------------------------------------------------
# ==============================================================================

class FadingText(object):
	def __init__(self, font, dim, pos):
		self.font = font
		self.dim = dim
		self.pos = pos
		self.seconds_left = 0
		self.surface = pygame.Surface(self.dim)

	def set_text(self, text, color=(255, 255, 255), seconds=2.0):
		text_texture = self.font.render(text, True, color)
		self.surface = pygame.Surface(self.dim)
		self.seconds_left = seconds
		self.surface.fill((0, 0, 0, 0))
		self.surface.blit(text_texture, (10, 11))

	def tick(self, _, clock):
		delta_seconds = 1e-3 * clock.get_time()
		self.seconds_left = max(0.0, self.seconds_left - delta_seconds)
		self.surface.set_alpha(500.0 * self.seconds_left)

	def render(self, display):
		display.blit(self.surface, self.pos)


# ==============================================================================
# -- CollisionSensor -----------------------------------------------------------
# ==============================================================================

class CollisionSensor(object):
	def __init__(self, world, actor, hud):
		self.sensor = None
		self._history = []
		self._actor = actor
		self._actor_mass = actor.get_physics_control().mass
		self._hud = hud
		self._world = world
		self.collision_flag = False
		self.n_seq = len(os.listdir(image_save_path))
		bp = self._world.get_blueprint_library().find('sensor.other.collision')
		self.sensor = self._world.spawn_actor(bp, carla.Transform(), attach_to=self._actor)
		# Pass the lambda a weak reference to self to avoid circular reference
		weak_self = weakref.ref(self)
		self.sensor.listen(lambda event: CollisionSensor._on_collision(weak_self, event)) 
		
		

	def get_collision_speeds(self):
		'''Convert collision intensities from momentem (kg*m/s) to speed (m/s).'''
		# print("collision_intensity")
		return [(c[0] , c[1] / self._actor_mass) for c in self._history]

	def get_collision_history(self):
		history = collections.defaultdict(int)
		for frame, intensity in self._history:
			history[frame] += intensity
		return history

	def delete_images(self, folder):
		# if collision, then delete specific images
		# else delete folder
		global collision_folder, safe_folder
		if folder == 'camera':
			imagestodelete = len(os.listdir(os.path.join(image_save_path , str(self.n_seq-1))))-seq_len
			if (self.collision_flag==True):
				collision_folder += 1
			else:
				safe_folder += 1

			if (self.collision_flag == True):
				for i in range(imagestodelete):
					os.remove(os.path.join(image_save_path , str(self.n_seq-1), '{}.npz'.format(i*2+1)))

			elif (self.collision_flag == False and collision_folder < safe_folder):
				# bigger than collision folder, then delete safe folder whole images
				safe_folder -= 1
				for i in range(len(os.listdir(os.path.join(image_save_path , str(self.n_seq-1))))):
					os.remove(os.path.join(image_save_path , str(self.n_seq-1), '{}.npz'.format(i*2+1)))

			elif (self.collision_flag == False):
				for i in range(imagestodelete):
					os.remove(os.path.join(image_save_path , str(self.n_seq-1), '{}.npz'.format(i*2+1)))

			print("Safe folder # : {}, Collision folder # : {}".format(safe_folder, collision_folder))

		elif folder == 'bev':
			imagestodelete = len(os.listdir(os.path.join(image_save_path_BEV , str(self.n_seq-1))))-seq_len
			if (self.collision_flag==True):
				collision_folder += 1
			else:
				safe_folder += 1

			if (self.collision_flag == True):
				for i in range(imagestodelete):
					os.remove(os.path.join(image_save_path_BEV , str(self.n_seq-1), '{}.npz'.format(i*2+1)))

			elif (self.collision_flag == False and collision_folder < safe_folder):
				# bigger than collision folder, then delete safe folder whole images
				safe_folder -= 1
				for i in range(len(os.listdir(os.path.join(image_save_path_BEV , str(self.n_seq-1))))):
					os.remove(os.path.join(image_save_path_BEV , str(self.n_seq-1), '{}.npz'.format(i*2+1)))

			elif (self.collision_flag == False):
				for i in range(imagestodelete):
					os.remove(os.path.join(image_save_path_BEV , str(self.n_seq-1), '{}.npz'.format(i*2+1)))

			print("Safe folder_BEV # : {}, Collision folder_BEV # : {}".format(safe_folder, collision_folder))
		

	@staticmethod
	def _on_collision(weak_self, event):
		global col_intensity
		self = weak_self()
		if not self:
			return
		actor_type = get_actor_display_name(event.other_actor)
		# if self._hud:
		# 	self._hud._notifications('Collision with %r, id = %d'
		# 							% (actor_type, event.other_actor.id))
		impulse = event.normal_impulse
		intensity = math.sqrt(impulse.x ** 2 + impulse.y ** 2 + impulse.z ** 2)
		# self._history.append((event.frame, intensity / self._actor_mass))
		# print("collision frame : {}".format(event.frame))
		
		col_intensity = intensity / self._actor_mass
		self.collision_flag = True
		print("colision intensity : {}".format(col_intensity))
		

	
	def destroy_sensor(self):
		if self.sensor is not None:
			global col_intensity
			col_intensity = 0
			self.sensor.stop()
			self.sensor.destroy()
			self.delete_images('camera')
			self.delete_images('bev')



# ==============================================================================
# -- CameraManager -------------------------------------------------------------
# ==============================================================================

class CameraManager(object):
	def __init__(self, world, actor, hud, camPosIndex):
		self.sensor = None
		self._surface = None
		self._actor = actor
		self._hud = hud
		self.images = []
		self._camera_transforms = [
			carla.Transform(carla.Location(x=-5.5, z=2.8), carla.Rotation(pitch=-15)),
			carla.Transform(carla.Location(x=0.45, z=1.66))]
		self._transform_index = camPosIndex
		self._sensors = [
			['sensor.camera.rgb', cc.Raw, 'Camera RGB'],
			['sensor.camera.depth', cc.Raw, 'Camera Depth (Raw)'],
			['sensor.camera.depth', cc.Depth, 'Camera Depth (Gray Scale)'],
			['sensor.camera.depth', cc.LogarithmicDepth, 'Camera Depth (Logarithmic Gray Scale)'],
			['sensor.camera.semantic_segmentation', cc.Raw, 'Camera Semantic Segmentation (Raw)'],
			['sensor.camera.semantic_segmentation', cc.CityScapesPalette,
			 'Camera Semantic Segmentation (CityScapes Palette)'],
			['sensor.lidar.ray_cast', None, 'Lidar (Ray-Cast)']]
		self._world = world
		bp_library = self._world.get_blueprint_library()
		for item in self._sensors:
			bp = bp_library.find(item[0])
			if item[0].startswith('sensor.camera'):
				bp.set_attribute('image_size_x', str(hud.dim[0]))
				bp.set_attribute('image_size_y', str(hud.dim[1]))
				bp.set_attribute('sensor_tick', '0.1')
			item.append(bp)
		self._index = None
		self.counter = 0
		self.n_seq = len(os.listdir(image_save_path))
		self.img_count = 1
		self.col_intensity_history = []
		if not os.path.exists(os.path.join(image_save_path,str(self.n_seq))):
			os.makedirs(os.path.join(image_save_path,str(self.n_seq)))

	def toggle_camera(self):
		set_transform((self._transform_index + 1) % len(self._camera_transforms))

	def set_transform(self, idx):
		self._transform_index = idx
		print(self._transform_index)
		self.sensor.set_transform(self._camera_transforms[self._transform_index])

	def set_sensor(self, index):
		index = index % len(self._sensors)
		needs_respawn = True if self._index is None \
			else self._sensors[index][0] != self._sensors[self._index][0]
		if needs_respawn:
			if self.sensor is not None:
				self.sensor.destroy()
				self._surface = None
			self.sensor = self._world.spawn_actor(
				self._sensors[index][-1],
				self._camera_transforms[self._transform_index],
				attach_to=self._actor)
			# Pass lambda a weak reference to self to avoid circular reference
			weak_self = weakref.ref(self)
			self.sensor.listen(lambda image: CameraManager._add_image(weak_self, image))
		self._index = index

	def render(self, display):
		if self._surface is not None:
			display.blit(self._surface, (0, 0))

	@staticmethod
	def _add_image(weak_self, image):
		self = weak_self()
		global col_intensity
		if not self:
			return
		if self._sensors[self._index][0].startswith('sensor.lidar'):
			points = np.frombuffer(image.raw_data, dtype=np.dtype('f4'))
			points = np.reshape(points, (int(points.shape[0] / 3), 3))
			lidar_data = np.array(points[:, :2])
			lidar_data *= min(self._hud.dim) / 100.0
			lidar_data += (0.5 * self._hud.dim[0], 0.5 * self._hud.dim[1])
			lidar_data = np.fabs(lidar_data)
			lidar_data = lidar_data.astype(np.int32)
			lidar_data = np.reshape(lidar_data, (-1, 2))
			lidar_img_size = (self._hud.dim[0], self._hud.dim[1], 3)
			lidar_img = np.zeros(lidar_img_size)
			lidar_img[tuple(lidar_data.T)] = (255, 255, 255)
			self._surface = pygame.surfarray.make_surface(lidar_img)
		else:
			image.convert(self._sensors[self._index][1])
			array = np.frombuffer(image.raw_data, dtype=np.dtype("uint8"))
			array = np.reshape(array, (image.height, image.width, 4))
			array = array[:, :, :3]
			array = array[:, :, ::-1]
			self._surface = pygame.surfarray.make_surface(array.swapaxes(0, 1))
		
		# image, collision save
		if (self.counter % 2) == 0: 
			save(array, self.img_count, self.n_seq, col_intensity, self._actor, 'camera')
			# self._world.tick()
		
		self.img_count += 1
		self.counter += 1
		# print("camera frame : {}".format(image.frame))
		# self.images.append(image)

	def destroy_sensor(self):
		if self.sensor is not None:
			self.sensor.stop()
			self.sensor.destroy()

class BevManager(object):
	def __init__(self, world, actor, client):
		self._world = world
		self._actor = actor
		self._client = client
		self.birdview_producer = BirdViewProducer(
			self._client,
			PixelDimensions(width=DEFAULT_WIDTH, height=DEFAULT_HEIGHT),
			render_lanes_on_junctions=False,
			pixels_per_meter=4,
			crop_type=BirdViewCropType.FRONT_AND_REAR_AREA,
		)
		self.counter = 0
		self.n_seq = len(os.listdir(image_save_path_BEV))
		self.img_count = 1
		if not os.path.exists(os.path.join(image_save_path_BEV,str(self.n_seq))):
			os.makedirs(os.path.join(image_save_path_BEV,str(self.n_seq)))

	def render(self):
		birdview: BirdView = self.birdview_producer.produce(agent_vehicle=self._actor)
		bgr_img = cv.cvtColor(BirdViewProducer.as_rgb(birdview), cv.COLOR_BGR2RGB)
		cv.imshow("BirdView RGB", bgr_img)
		bgr_img = bgr_img[:,:,:3][:]
		if ((self.counter % 2) == 0):
			save(bgr_img, self.img_count, self.n_seq, col_intensity, self._actor, 'bev')
		
		self.img_count += 1
		self.counter += 1

	def destry(self):
		cv.destroyAllWindows()
