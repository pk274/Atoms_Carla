import carla
from ATOMs_Analysis.atoms_config import ExperimentConfig as conf

class WeatherOrganizer(object):
    def __init__(self, macro_weather: str):
        pass
        

    def set_weather(self, world):
        if conf.WEATHER == "sunny":
            weather = carla.WeatherParameters(5.0, 0.0, 0.0, 10.0, -1.0, 45.0, 2.0, 0.75, 0.1, 0.0, 1.0, 0.03, 0.0331, 0.0 )
        elif conf.WEATHER == "cloudy":
            weather = carla.WeatherParameters(60.0, 0.0, 0.0, 10.0, -1.0, 45.0, 3.0, 0.75, 0.1, 0.0, 1.0, 0.03, 0.0331, 0.0 )
        elif conf.WEATHER == "night":
            weather = carla.WeatherParameters(5.0, 0.0, 50.0, 10.0, -1.0, -90.0, 60.0, 75.0, 1.0, 60.0, 1.0, 0.03, 0.0331, 0.0)
        elif conf.WEATHER == "rainy":
            weather = carla.WeatherParameters(100.0, 100.0, 90.0, 100.0, -1.0, 45.0, 7.0, 0.75, 0.1, 0.0, 1.0, 0.03, 0.0331, 0.0)
        elif conf.WEATHER == "foggy":
            weather = carla.WeatherParameters(80.0, 0.0, 0.0, 1.0, 40.0, 45.0, 30.0, 4.0, 0.8, 10.0, 1.0, 0.03, 0.0331, 0.0)
        else:
            print("No valid weather given")
        world.set_weather(weather)

    def __str__(self):
        return '%s %s' % (self._sun, self._storm)


