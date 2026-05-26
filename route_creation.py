from PCLA import route_maker, location_to_waypoint
import carla

client = carla.Client('localhost', 2000)
world = client.get_world()

vehicle_spawn_points = world.get_map().get_spawn_points()
startLoc = vehicle_spawn_points[398].location
middleLoc1 = vehicle_spawn_points[287].location
middleLoc2 = vehicle_spawn_points[312].location
endLoc = vehicle_spawn_points[368].location

# 1. Generate waypoints
waypoints1 = location_to_waypoint(client, startLoc, middleLoc1)
waypoints2 = location_to_waypoint(client, middleLoc2, endLoc)

waypoints = waypoints1 + waypoints2

# 2. Create the PCLA XML route file
route_maker(waypoints, "route.xml")