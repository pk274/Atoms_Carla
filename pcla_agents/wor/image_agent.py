import os
import math
import yaml
import lmdb
import numpy as np
import torch
import wandb
import carla
import random
import string

from torch.distributions.categorical import Categorical

from leaderboard_codes.autonomous_agent1 import AutonomousAgent, Track
from pcla_agents.wor.utils.visualization import visualize_obs
from ATOMs_Analysis.perturbation_manager import PerturbationManager
from ATOMs_Analysis.atoms_config import ExperimentConfig as conf
from ATOMs_Analysis.saliency.lrp_analysis import LRPCameraModel
from ATOMs_Analysis.utils.visualization_carla import visualize_relevance, visualize_segmentation
from ATOMs_Analysis.saliency.atoms_carla import ATOMsCarla
from ATOMs_Analysis.detection.baseline_dataset import BaselineDataCollector
from ATOMs_Analysis.detection.dataset import TestDataCollector

from pcla_agents.wor.rails.models import EgoModel, CameraModel
from pcla_agents.wor.waypointer import Waypointer

def get_entry_point():
    return 'ImageAgent'

class ImageAgent(AutonomousAgent):
    
    """
    Trained image agent
    """
    
    def setup(self, path_to_conf_file):
        """
        Setup the agent parameters
        """

        self.track = Track.SENSORS
        self.num_frames = 0

        config_dir = os.path.dirname(os.path.abspath(path_to_conf_file))

        with open(path_to_conf_file, 'r') as f:
            config = yaml.safe_load(f)

        # Loop through the items and correct the paths
        for key, value in config.items():
            if key.endswith('_dir'):
                # The paths in your YAML file are relative to the PCLA folder.
                # We need to go up two directories from the config folder to get to PCLA.
                
                project_root = os.path.dirname(os.path.dirname(config_dir))
                
                # Construct the absolute path by joining the project root and the relative path from the YAML
                absolute_path = os.path.join(project_root, value)
                
                # Now, set the attribute with the corrected absolute path
                setattr(self, key, absolute_path)
            else:
                # For non-path values, set them as is
                setattr(self, key, value)

        self.device = torch.device('cpu')

        self.image_model = CameraModel(config).to(self.device)
        self.image_model.load_state_dict(torch.load(self.main_model_dir, map_location=torch.device('cpu')))
        self.image_model.eval()
        for param in self.image_model.parameters():
            param.requires_grad = False

        self.vizs = []

        self.waypointer = None

        if self.log_wandb:
            wandb.init(project='carla_evaluate')
            
        self.steers = torch.tensor(np.linspace(-self.max_steers,self.max_steers,self.num_steers)).float().to(self.device)
        self.throts = torch.tensor(np.linspace(0,self.max_throts,self.num_throts)).float().to(self.device)

        self.prev_steer = 0
        self.lane_change_counter = 0
        self.stop_counter = 0

        self.pm = PerturbationManager(verbose=False)
        self.lrp = LRPCameraModel(self.image_model)
        self.data_collector = BaselineDataCollector(conf.IMAGE_SAMPLE_INTERVAL)
        self.test_data_collector = TestDataCollector(conf.TEST_SAMPLE_INTERVAL, perturbation_name=conf.PERTURBATION)

    def destroy(self):
        if len(self.vizs) == 0:
            return

        self.flush_data()
        self.prev_steer = 0
        self.lane_change_counter = 0
        self.stop_counter = 0
        self.lane_changed = None
        
        del self.waypointer
        del self.image_model
    
    def flush_data(self):

        if self.log_wandb:
            wandb.log({
                'vid': wandb.Video(np.stack(self.vizs).transpose((0,3,1,2)), fps=20, format='mp4')
            })
            
        self.vizs.clear()

    def sensors(self):
        sensors = [ # Sensors modified to match online leaderboard based on https://github.com/dotchen/WorldOnRails/issues/27
            {'type': 'sensor.speedometer', 'id': 'EGO'},
            {'type': 'sensor.other.gnss', 'x': 0., 'y': 0.0, 'z': self.camera_z, 'id': 'GPS'},
            {'type': 'sensor.camera.rgb', 'x': self.camera_x, 'y': 0, 'z': self.camera_z, 'roll': 0.0, 'pitch': 0.0, 'yaw': -55.0,
            'width': 160, 'height': 240, 'fov': 60, 'id': f'Wide_RGB_0'},
            {'type': 'sensor.camera.rgb', 'x': self.camera_x, 'y': 0, 'z': self.camera_z, 'roll': 0.0, 'pitch': 0.0, 'yaw': 0.0,
            'width': 160, 'height': 240, 'fov': 60, 'id': f'Wide_RGB_1'},
            {'type': 'sensor.camera.rgb', 'x': self.camera_x, 'y': 0, 'z': self.camera_z, 'roll': 0.0, 'pitch': 0.0, 'yaw':  55.0,
            'width': 160, 'height': 240, 'fov': 60, 'id': f'Wide_RGB_2'},
            {'type': 'sensor.camera.rgb', 'x': self.camera_x, 'y': 0, 'z': self.camera_z, 'roll': 0.0, 'pitch': 0.0, 'yaw': 0.0,
            'width': 384, 'height': 240, 'fov': 50, 'id': f'Narrow_RGB'},

            # Additional semantic cameras for ATOMs labeling
            {'type': 'sensor.camera.semantic_segmentation', 'x': self.camera_x, 'y': 0, 'z': self.camera_z, 'roll': 0.0, 'pitch': 0.0, 'yaw': -55.0,
            'width': 160, 'height': 240, 'fov': 60, 'id': f'Wide_Semantic_0'},
            {'type': 'sensor.camera.semantic_segmentation', 'x': self.camera_x, 'y': 0, 'z': self.camera_z, 'roll': 0.0, 'pitch': 0.0, 'yaw': 0.0,
            'width': 160, 'height': 240, 'fov': 60, 'id': f'Wide_Semantic_1'},
            {'type': 'sensor.camera.semantic_segmentation', 'x': self.camera_x, 'y': 0, 'z': self.camera_z, 'roll': 0.0, 'pitch': 0.0, 'yaw':  55.0,
            'width': 160, 'height': 240, 'fov': 60, 'id': f'Wide_Semantic_2'},
            {'type': 'sensor.camera.semantic_segmentation', 'x': self.camera_x, 'y': 0, 'z': self.camera_z, 'roll': 0.0, 'pitch': 0.0, 'yaw': 0.0,
            'width': 384, 'height': 240, 'fov': 50, 'id': f'Narrow_Semantic'}
        ]
        return sensors

    def run_step(self, input_data, timestamp, vehicle=None):
        # changed to match the online leaderboard based on https://github.com/dotchen/WorldOnRails/issues/27
        wide_rgbs = []
        wide_sems = []
        for i in range(3):
            # RGB
            _, wide_rgb = input_data.get(f'Wide_RGB_{i}')
            wide_rgb_crop = wide_rgb[self.wide_crop_top:,:,:3]
            _wide_rgb = wide_rgb_crop[...,::-1].copy()
            wide_rgbs.append(_wide_rgb)

            # Semantic
            _, wide_sem = input_data.get(f'Wide_Semantic_{i}')
            wide_sem_crop = wide_sem[self.wide_crop_top:, :, 2]  # shape: (H, W)
            wide_sems.append(wide_sem_crop)
        
        _, narr_rgb = input_data.get(f'Narrow_RGB')
        _, narr_sem = input_data.get(f'Narrow_Semantic')
        _, ego = input_data.get('EGO')
        _, gps = input_data.get('GPS')
        spd = ego.get('speed')

        if self.waypointer is None:
            self.waypointer = Waypointer(self._global_plan, gps)

        _, _, cmd = self.waypointer.tick(gps)

        if timestamp >= conf.INJECTION_TIME:
            # Problematic perturbations: Right camera loss, brightness scale <= 0.2, b scale >= 4
            if timestamp <= conf.INJECTION_TIME + 0.1:
                print("!!! PERTURBATION ACTIVATED !!!")
            if conf.PERTURBATION != "fgsm" and conf.PERTURBATION != "pgd":
                wide_rgbs = self.pm.perturb_wide_image(wide_rgbs, perturbation=conf.PERTURBATION,
                                                       intensity=conf.INTENSITY, camera_index=conf.CAM_INDEX)
                narr_rgb = self.pm.perturb_narrow_image(narr_rgb, perturbation=conf.PERTURBATION,
                                                        intensity=conf.INTENSITY)

        # This is of shape [192, 480, 3] and has the format BGR !!!!
        wide_rgbs_con = np.concatenate([wide_rgbs[0],wide_rgbs[1],wide_rgbs[2]], axis=1)
        wide_sems_con = np.concatenate([wide_sems[0],wide_sems[1],wide_sems[2]], axis=1)
        rgb = np.concatenate([wide_rgbs_con[..., ::-1].copy()], axis=1) #BGR -> RGB
        #rgb = np.concatenate([narr_rgb[...,:3]], axis=1)
        narr_rgb_crop = narr_rgb[:-self.narr_crop_bottom,:,:3]
        narr_sem_crop = narr_sem[:-self.narr_crop_bottom, :, 2]
        _narr_rgb = narr_rgb_crop[...,::-1].copy()

        
        cmd_value = cmd.value-1
        cmd_value = 3 if cmd_value < 0 else cmd_value

        if cmd_value in [4,5]:
            if self.lane_changed is not None and cmd_value != self.lane_changed:
                self.lane_change_counter = 0

            self.lane_change_counter += 1
            self.lane_changed = cmd_value if self.lane_change_counter > {4:200,5:200}.get(cmd_value) else None
        else:
            self.lane_change_counter = 0
            self.lane_changed = None

        if cmd_value == self.lane_changed:
            cmd_value = 3

        # This turns it into shape [3, 192, 480]
        wide_rgbs_ = torch.tensor(wide_rgbs_con[None]).float().permute(0,3,1,2).to(self.device)
        _wide_rgb = torch.tensor(_wide_rgb[None]).float().permute(0,3,1,2).to(self.device)
        _narr_rgb = torch.tensor(_narr_rgb[None]).float().permute(0,3,1,2).to(self.device)

        
        #FSGM Injection
        if conf.PERTURBATION == "fgsm" and timestamp >= conf.INJECTION_TIME:
                wide_rgbs_, _narr_rgb = self.pm.fgsm_attack(self.image_model, wide_rgbs_, _narr_rgb,
                                                cmd_value, target="steer_right", epsilon=conf.EPSILON, apply_to_narrow=False)
                rgb = wide_rgbs_.detach().cpu().squeeze(0).permute(1, 2, 0).numpy()
                rgb = rgb.clip(0, 255).astype(np.uint8)
                rgb = rgb[..., ::-1]  # only if needed (BGR → RGB)
        elif conf.PERTURBATION == "pgd" and timestamp >= conf.INJECTION_TIME:
                wide_rgbs_, _narr_rgb = self.pm.pgd_attack(self.image_model, wide_rgbs_, _narr_rgb,
                                                cmd_value, target="max_steer", epsilon=conf.EPSILON, apply_to_narrow=False)
                rgb = wide_rgbs_.detach().cpu().squeeze(0).permute(1, 2, 0).numpy()
                rgb = rgb.clip(0, 255).astype(np.uint8)
                rgb = rgb[..., ::-1]  # only if needed (BGR → RGB)


        if self.all_speeds:
            with torch.no_grad():
                steer_logits, throt_logits, brake_logits = self.image_model.policy(wide_rgbs_, _narr_rgb, cmd_value)
                #if timestamp % 1 < 0.05:
                #    self.lrp.update_context(wide_rgbs_, narr_rgb, spd)
                #    wide_rel, narr_rel, wide_frac, is_brake = self.lrp.forward_relevance(wide_rgbs_, _narr_rgb,
                #                                                                        beg="fc", end="input", cmd=cmd_value)
                #    visualize_relevance(narr_rel)
            # Interpolate logits
            steer_logit = self._lerp(steer_logits, spd)
            throt_logit = self._lerp(throt_logits, spd)
            brake_logit = self._lerp(brake_logits, spd)
        else:
            with torch.no_grad():
                steer_logit, throt_logit, brake_logit = self.image_model.policy(_wide_rgb, _narr_rgb, cmd_value, spd=torch.tensor([spd]).float().to(self.device))

        
        action_prob = self.action_prob(steer_logit, throt_logit, brake_logit)

        brake_prob = float(action_prob[-1])

        steer = float(self.steers @ torch.softmax(steer_logit, dim=0))
        throt = float(self.throts @ torch.softmax(throt_logit, dim=0))

        steer, throt, brake = self.post_process(steer, throt, brake_prob, spd, cmd_value)

        # Add frame to baseline if BASELINE_COLLECTION is True
        if conf.BASELINE_RECORDING_MODE:
            self.data_collector.add_frame(wide_rgbs_, _narr_rgb, wide_sems_con, narr_sem_crop, cmd_value, spd, is_brake=bool(brake))
        elif conf.TESTSET_RECORDING_MODE:
            self.test_data_collector.add_frame(wide_rgbs_, _narr_rgb, wide_sems_con, narr_sem_crop, cmd_value, spd, is_brake=bool(brake))
        elif conf.LIVE_PERTURBATION_RECORDING_MODE:
            self.test_data_collector.add_frame(wide_rgbs_, _narr_rgb, wide_sems_con,
                                               narr_sem_crop, cmd_value, spd, is_brake=bool(brake),
                                               live_perturbation=True,
                                               is_perturbed=(timestamp >= conf.INJECTION_TIME))

        self.vizs.append(visualize_obs(rgb, 0, (steer, throt, brake), spd, cmd=cmd_value+1))

        if len(self.vizs) > 1000:
            self.flush_data()

        self.num_frames += 1

        return carla.VehicleControl(steer=steer, throttle=throt, brake=brake)
    
    
    def _lerp(self, v, x):
        D = v.shape[0]

        min_val = self.min_speeds
        max_val = self.max_speeds

        x = (x - min_val)/(max_val - min_val)*(D-1)

        x0, x1 = max(min(math.floor(x), D-1),0), max(min(math.ceil(x), D-1),0)
        w = x - x0

        return (1-w) * v[x0] + w * v[x1]

    def action_prob(self, steer_logit, throt_logit, brake_logit):

        steer_logit = steer_logit.repeat(self.num_throts)
        throt_logit = throt_logit.repeat_interleave(self.num_steers)

        action_logit = torch.cat([steer_logit, throt_logit, brake_logit[None]])

        return torch.softmax(action_logit, dim=0)

    def post_process(self, steer, throt, brake_prob, spd, cmd):
        
        if brake_prob > 0.5:
            steer, throt, brake = 0, 0, 1
        else:
            brake = 0
            if conf.HIGH_SPEED_MODE:
                throt = max(0.6, throt)
            else:
                throt = max(0.4, throt)

        # # To compensate for non-linearity of throttle<->acceleration
        # if throt > 0.1 and throt < 0.4:
        #     throt = 0.4
        # elif throt < 0.1 and brake_prob > 0.3:
        #     brake = 1
        max_spd = {0:10,1:10}.get(cmd, 20)/3.6
        if conf.SPEED_MODE:
            max_spd = {0:10,1:10}.get(cmd, 40)/3.6  # 50 for skipping traffic lights :)
        if conf.HIGH_SPEED_MODE:
            max_spd = {0:10,1:10}.get(cmd, 100)/3.6
        if spd > max_spd: # 10 km/h for turning, 15km/h elsewhere
            throt = 0

        # if cmd == 2:
        #     steer = min(max(steer, -0.2), 0.2)

        # if cmd in [4,5]:
        #     steer = min(max(steer, -0.4), 0.4) # no crazy steerings when lane changing

        return steer, throt, brake
    
def load_state_dict(model, path):

    from collections import OrderedDict
    new_state_dict = OrderedDict()
    state_dict = torch.load(path, map_location=torch.device('cpu'))
    
    for k, v in state_dict.items():
        name = k[7:] # remove `module.`
        new_state_dict[name] = v
    
    model.load_state_dict(new_state_dict)
