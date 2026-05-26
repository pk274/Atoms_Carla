#!/usr/bin/env python

# Copyright (c) 2021 Computer Vision Center (CVC) at the Universitat Autonoma de
# Barcelona (UAB).
#
# This work is licensed under the terms of the MIT license.
# For a copy, see <https://opensource.org/licenses/MIT>.

"""Example script to generate traffic in the simulation"""

import carla

from carla import VehicleLightState as vls
from carla.command import SpawnActor, SetAutopilot, FutureActor, DestroyActor

import argparse
import logging
from numpy import random
import time

from ATOMs_Analysis.atoms_config import ExperimentConfig as conf

class TrafficOrganizer:

    def __init__(self, client):
        self.client = client
        self.world = self.client.get_world()
        self.seed = 4
        self.tm_port = 8000
        self.respawn = False
        self.hybrid = False
        self.asynch = False
        self.filterv = 'vehicle.*'
        self.generationv = 'ALL'
        self.filterw = 'walker.pedestrian.*'
        self.generationw = '2'
        self.safe = True
        self.number_of_vehicles = 20
        self.hero = False
        self.car_lights_on = False
        if conf.WEATHER == "foggy" or conf.WEATHER == "night":
            self.car_lights_on = True
        self.number_of_walkers = 15
        self.synchronous_master = True
        self.vehicles_list = None
        self.all_id = None
        self.all_actors = None
        self.spawn_manually = conf.MANUAL_SPAWNS
        self.manual_spwan_points = [155, 272, 144, 234, 163, 179]

        self.traffic_manager = self.client.get_trafficmanager(self.tm_port)




    def get_actor_blueprints(self, world, filter, generation):
        bps = world.get_blueprint_library().filter(filter)

        if generation.lower() == "all":
            return bps

        # If the filter returns only one bp, we assume that this one needed
        # and therefore, we ignore the generation
        if len(bps) == 1:
            return bps

        try:
            int_generation = int(generation)
            # Check if generation is in available generations
            if int_generation in [1, 2, 3]:
                bps = [x for x in bps if int(x.get_attribute('generation')) == int_generation]
                return bps
            else:
                print("   Warning! Actor Generation is not valid. No actor will be spawned.")
                return []
        except:
            print("   Warning! Actor Generation is not valid. No actor will be spawned.")
            return []

    def generate_traffic(self):


        logging.basicConfig(format='%(levelname)s: %(message)s', level=logging.INFO)

        self.vehicles_list = []
        self.walkers_list = []
        self.all_id = []
        synchronous_master = False
        random.seed(self.seed if self.seed is not None else int(time.time()))

        self.traffic_manager.set_global_distance_to_leading_vehicle(2.5)
        if self.respawn:
            self.traffic_manager.set_respawn_dormant_vehicles(True)
        if self.hybrid:
            self.traffic_manager.set_hybrid_physics_mode(True)
            self.traffic_manager.set_hybrid_physics_radius(70.0)
        if self.seed is not None:
            self.traffic_manager.set_random_device_seed(self.seed)
        settings = self.world.get_settings()
        if not self.asynch:
            self.traffic_manager.set_synchronous_mode(True)
            if not settings.synchronous_mode:
                self.synchronous_master = True
                settings.synchronous_mode = True
                settings.fixed_delta_seconds = 0.05
            else:
                self.synchronous_master = False
        else:
            print("You are currently in asynchronous mode. If this is a traffic simulation, \
            you could experience some issues. If it's not working correctly, switch to synchronous \
            mode by using traffic_manager.set_synchronous_mode(True)")
        self.world.apply_settings(settings)
        blueprints = self.get_actor_blueprints(self.world, self.filterv, self.generationv)
        if not blueprints:
            raise ValueError("Couldn't find any vehicles with the specified filters")
        blueprintsWalkers = self.get_actor_blueprints(self.world, self.filterw, self.generationw)
        if not blueprintsWalkers:
            raise ValueError("Couldn't find any walkers with the specified filters")
        if self.safe:
            blueprints = [x for x in blueprints if x.get_attribute('base_type') == 'car']
        blueprints = sorted(blueprints, key=lambda bp: bp.id)
        spawn_points = self.world.get_map().get_spawn_points()
        spawn_points.pop(conf.SPAWN_INDEX)
        number_of_spawn_points = len(spawn_points)
        if self.number_of_vehicles < number_of_spawn_points:
            random.shuffle(spawn_points)
        elif self.number_of_vehicles > number_of_spawn_points:
            msg = 'requested %d vehicles, but could only find %d spawn points'
            logging.warning(msg, self.number_of_vehicles, number_of_spawn_points)
            self.number_of_vehicles = number_of_spawn_points
        # --------------
        # Spawn vehicles
        # --------------
        batch = []
        hero = self.hero
        if self.spawn_manually:
            manual_spawns = []
            for i in self.manual_spwan_points:
                manual_spawns.append(self.world.get_map().get_spawn_points()[i])
            for n, transform in enumerate(manual_spawns):
                if n >= self.number_of_vehicles:
                    break
                blueprint = random.choice(blueprints)
                if blueprint.has_attribute('color'):
                    color = random.choice(blueprint.get_attribute('color').recommended_values)
                    blueprint.set_attribute('color', color)
                if blueprint.has_attribute('driver_id'):
                    driver_id = random.choice(blueprint.get_attribute('driver_id').recommended_values)
                    blueprint.set_attribute('driver_id', driver_id)
                if hero:
                    blueprint.set_attribute('role_name', 'hero')
                    hero = False
                else:
                    blueprint.set_attribute('role_name', 'autopilot')
                # spawn the cars and set their autopilot and light state all together
                batch.append(SpawnActor(blueprint, transform)
                    .then(SetAutopilot(FutureActor, True, self.traffic_manager.get_port())))
        for n, transform in enumerate(spawn_points):
            if n >= self.number_of_vehicles:
                break
            blueprint = random.choice(blueprints)
            if blueprint.has_attribute('color'):
                color = random.choice(blueprint.get_attribute('color').recommended_values)
                blueprint.set_attribute('color', color)
            if blueprint.has_attribute('driver_id'):
                driver_id = random.choice(blueprint.get_attribute('driver_id').recommended_values)
                blueprint.set_attribute('driver_id', driver_id)
            if hero:
                blueprint.set_attribute('role_name', 'hero')
                hero = False
            else:
                blueprint.set_attribute('role_name', 'autopilot')
            # spawn the cars and set their autopilot and light state all together
            batch.append(SpawnActor(blueprint, transform)
                .then(SetAutopilot(FutureActor, True, self.traffic_manager.get_port())))
        for response in self.client.apply_batch_sync(batch, synchronous_master):
            if response.error:
                logging.error(response.error)
            else:
                self.vehicles_list.append(response.actor_id)
        # Set automatic vehicle lights update if specified
        if self.car_lights_on:
            all_vehicle_actors = self.world.get_actors(self.vehicles_list)
            for actor in all_vehicle_actors:
                self.traffic_manager.update_vehicle_lights(actor, True)
        # -------------
        # Spawn Walkers
        # -------------
        # some settings
        percentagePedestriansRunning = 0.0      # how many pedestrians will run
        percentagePedestriansCrossing = 0.0     # how many pedestrians will walk through the road
        if self.seed:
            self.world.set_pedestrians_seed(self.seed)
            random.seed(self.seed)
        # 1. take all the random locations to spawn
        spawn_points = []
        for i in range(self.number_of_walkers):
            spawn_point = carla.Transform()
            loc = self.world.get_random_location_from_navigation()
            if (loc != None):
                spawn_point.location = loc
                spawn_points.append(spawn_point)
        # 2. we spawn the walker object
        batch = []
        walker_speed = []
        for spawn_point in spawn_points:
            walker_bp = random.choice(blueprintsWalkers)
            # set as not invincible
            probability = random.randint(0,100 + 1);
            if walker_bp.has_attribute('is_invincible'):
                walker_bp.set_attribute('is_invincible', 'false')
            if walker_bp.has_attribute('can_use_wheelchair') and probability < 11:
                walker_bp.set_attribute('use_wheelchair', 'true')
            # set the max speed
            if walker_bp.has_attribute('speed'):
                if (random.random() > percentagePedestriansRunning):
                    # walking
                    walker_speed.append(walker_bp.get_attribute('speed').recommended_values[1])
                else:
                    # running
                    walker_speed.append(walker_bp.get_attribute('speed').recommended_values[2])
            else:
                print("Walker has no speed")
                walker_speed.append(0.0)
            batch.append(SpawnActor(walker_bp, spawn_point))
        results = self.client.apply_batch_sync(batch, True)
        walker_speed2 = []
        for i in range(len(results)):
            if results[i].error:
                logging.error(results[i].error)
            else:
                self.walkers_list.append({"id": results[i].actor_id})
                walker_speed2.append(walker_speed[i])
        walker_speed = walker_speed2
        # 3. we spawn the walker controller
        batch = []
        walker_controller_bp = self.world.get_blueprint_library().find('controller.ai.walker')
        for i in range(len(self.walkers_list)):
            batch.append(SpawnActor(walker_controller_bp, carla.Transform(), self.walkers_list[i]["id"]))
        results = self.client.apply_batch_sync(batch, True)
        for i in range(len(results)):
            if results[i].error:
                logging.error(results[i].error)
            else:
                self.walkers_list[i]["con"] = results[i].actor_id
        # 4. we put together the walkers and controllers id to get the objects from their id
        for i in range(len(self.walkers_list)):
            self.all_id.append(self.walkers_list[i]["con"])
            self.all_id.append(self.walkers_list[i]["id"])
        self.all_actors = self.world.get_actors(self.all_id)
        # wait for a tick to ensure client receives the last transform of the walkers we have just created
        self.world.tick()
        # 5. initialize each controller and set target to walk to (list is [controler, actor, controller, actor ...])
        # set how many pedestrians can cross the road
        self.world.set_pedestrians_cross_factor(percentagePedestriansCrossing)
        for i in range(0, len(self.all_id), 2):
            # start walker
            self.all_actors[i].start()
            # set walk to random point
            self.all_actors[i].go_to_location(self.world.get_random_location_from_navigation())
            # max speed
            self.all_actors[i].set_max_speed(float(walker_speed[int(i/2)]))
        print('spawned %d vehicles and %d walkers, press Ctrl+C to exit.' % (len(self.vehicles_list), len(self.walkers_list)))
        # Example of how to use Traffic Manager parameters
        self.traffic_manager.global_percentage_speed_difference(30.0)


    def clean_up(self):

        if not self.asynch and self.synchronous_master:
            settings = self.world.get_settings()
            settings.synchronous_mode = False
            settings.no_rendering_mode = False
            settings.fixed_delta_seconds = None
            self.world.apply_settings(settings)
        print('\ndestroying %d vehicles' % len(self.vehicles_list))
        self.client.apply_batch([DestroyActor(x) for x in self.vehicles_list])
        # stop walker controllers (list is [controller, actor, controller, actor ...])
        for i in range(0, len(self.all_id), 2):
            self.all_actors[i].stop()
        print('\ndestroying %d walkers' % len(self.walkers_list))
        self.client.apply_batch([DestroyActor(x) for x in self.all_id])
        time.sleep(0.5)

