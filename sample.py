import carla
import time
from PCLA import PCLA
import sys

from ATOMs_Analysis.atoms_config import ExperimentConfig as conf
from map_manupulation.generate_traffic import TrafficOrganizer
from map_manupulation.dynamic_weather import WeatherOrganizer


def main():

    HOST_IP  = "localhost"
    client = carla.Client(HOST_IP, 2000)
    client.set_timeout(10.0)
    client.load_world(conf.TOWN)
    synchronous_master = False
    pcla = None
    settings = None
    to = TrafficOrganizer(client)
    wo = WeatherOrganizer(conf.WEATHER)

    try:
        world = client.get_world()

        to.generate_traffic()
        wo.set_weather(world)

        settings = world.get_settings()
        
        #settings.no_rendering_mode = True
        world.apply_settings(settings)
        
        # Finding actors
        bpLibrary = world.get_blueprint_library()

        ## Finding vehicle
        vehicleBP = bpLibrary.filter('model3')[0]

        vehicle_spawn_points = world.get_map().get_spawn_points()

        ### Spawn vehicle
        vehicle = world.spawn_actor(vehicleBP, vehicle_spawn_points[conf.SPAWN_INDEX])
        if conf.WEATHER == "night" or conf.WEATHER ==  "foggy":
            vehicle.set_light_state(carla.VehicleLightState(carla.VehicleLightState.All))
        
        # Retrieve the spectator object
        spectator = world.get_spectator()

        # Set the spectator with our transform
        spectator.set_transform(carla.Transform(carla.Location(x=conf.SPEC_POS[0], y=conf.SPEC_POS[1], z=conf.SPEC_POS[2]),
                                                carla.Rotation(pitch=conf.SPEC_ROT[0], yaw=conf.SPEC_ROT[1], roll=conf.SPEC_ROT[2])))

        world.tick()

        if conf.AGENT == "LBC":
            agent = "lbc_lb"
        elif conf.AGENT == "TFV6" and conf.LIVE_PERTURBATION_RECORDING_MODE:
            agent = "tfv6_livepert"
        elif conf.AGENT == "TFV6":
            agent = "tfv6_datacollect"
        else:
            agent = "wor_lb"
        route = f"./route_{conf.TOWN}.xml"
        pcla = PCLA(agent, vehicle, route, client)
        
        print('\nSpawned the vehicle with model =', agent,', press Ctrl+C to exit.\n')
        step = 0
        while True:
            try:
                ego_action = pcla.get_action()
                vehicle.apply_control(ego_action)
                world.tick()
                step += 1
            except Exception as e:
                print(f'\nError at step {step}:')
                print(f'{type(e).__name__}: {e}\n')
                import traceback
                traceback.print_exc()
                break
    
    finally:
        #settings.no_rendering_mode = False
        if settings is not None:
            settings.synchronous_mode = False
            world.apply_settings(settings)

        # Destroy vehicle
        print('\nCleaning up the vehicle')
        if pcla is not None:
            pcla.cleanup()
        time.sleep(0.5)
        to.clean_up()

if __name__ == '__main__':

    try:
        main()
    except KeyboardInterrupt:
        pass
    finally:
        print('Done.')
