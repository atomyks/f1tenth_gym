# MIT License

# Copyright (c) 2020 Joseph Auckley, Matthew O'Kelly, Aman Sinha, Hongrui Zheng

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

'''
Author: Hongrui Zheng
'''

# gym imports
import gym
from gym import error, spaces, utils
from gym.utils import seeding

# base classes
from f110_gym.envs.base_classes import Simulator

# others
import numpy as np
import os
import time

# gl
import pyglet
pyglet.options['debug_gl'] = False
from pyglet import gl

# constants

# rendering
VIDEO_W = 600
VIDEO_H = 400
WINDOW_W = 1000
WINDOW_H = 800

class F110Env(gym.Env):
    """
    OpenAI gym environment for F1TENTH
    
    Env should be initialized by calling gym.make('f110_gym:f110-v0', **kwargs)

    Args:
        kwargs:
            seed (int, default=12345): seed for random state and reproducibility
            
            map (str, default='vegas'): name of the map used for the environment. Currently, available environments include: 'berlin', 'vegas', 'skirk'. You could use a string of the absolute path to the yaml file of your custom map.
        
            map_ext (str, default='png'): image extension of the map image file. For example 'png', 'pgm'

            model (str, default='ST'): vehicle model to use. Options: 'dynamic_ST' - dynamic single track model, 'MB' - multi body model
        
            params (dict, default={'mu': 1.0489, 'C_Sf':, 'C_Sr':, 'lf': 0.15875, 'lr': 0.17145, 'h': 0.074, 'm': 3.74, 'I': 0.04712, 's_min': -0.4189, 's_max': 0.4189, 'sv_min': -3.2, 'sv_max': 3.2, 'v_switch':7.319, 'a_max': 9.51, 'v_min':-5.0, 'v_max': 20.0, 'width': 0.31, 'length': 0.58}): dictionary of vehicle parameters.
            mu: surface friction coefficient
            C_Sf: Cornering stiffness coefficient, front
            C_Sr: Cornering stiffness coefficient, rear
            lf: Distance from center of gravity to front axle
            lr: Distance from center of gravity to rear axle
            h: Height of center of gravity
            m: Total mass of the vehicle
            I: Moment of inertial of the entire vehicle about the z axis
            s_min: Minimum steering angle constraint
            s_max: Maximum steering angle constraint
            sv_min: Minimum steering velocity constraint
            sv_max: Maximum steering velocity constraint
            v_switch: Switching velocity (velocity at which the acceleration is no longer able to create wheel spin)
            a_max: Maximum longitudinal acceleration
            v_min: Minimum longitudinal velocity
            v_max: Maximum longitudinal velocity
            width: width of the vehicle in meters
            length: length of the vehicle in meters

            num_agents (int, default=2): number of agents in the environment

            drive_control_mode (str, default='v'): Drive command. Options: 'vel' - velocity, 'acc' - acceleration

            timestep (float, default=0.01): physics timestep

            ego_idx (int, default=0): ego's index in list of agents
    """
    metadata = {'render.modes': ['human', 'human_fast']}

    # rendering
    renderer = None
    current_obs = None
    render_callbacks = []

    def __init__(self, **kwargs):        
        # kwargs extraction
        try:
            self.seed = kwargs['seed']
        except:
            self.seed = 12345
        try:
            self.map_name = kwargs['map']
            # different default maps
            if self.map_name == 'berlin':
                self.map_path = os.path.dirname(os.path.abspath(__file__)) + '/maps/berlin.yaml'
            elif self.map_name == 'skirk':
                self.map_path = os.path.dirname(os.path.abspath(__file__)) + '/maps/skirk.yaml'
            elif self.map_name == 'levine':
                self.map_path = os.path.dirname(os.path.abspath(__file__)) + '/maps/levine.yaml'
            else:
                self.map_path = self.map_name + '.yaml'
        except:
            self.map_path = os.path.dirname(os.path.abspath(__file__)) + '/maps/vegas.yaml'

        try:
            self.map_ext = kwargs['map_ext']
        except:
            self.map_ext = '.png'

        try:
            self.model = kwargs['model']
        except:
            self.model = 'dynamic_ST'
        # check valid options
        assert self.model in ['dynamic_ST', 'MB']

        try:
            self.params = kwargs['params']
        except:
            if self.model == 'dynamic_ST':
                # self.params = {'mu': 1.0489, 'C_Sf': 4.718, 'C_Sr': 5.4562, 'lf': 0.15875, 'lr': 0.17145, 'h': 0.074,
                #                'm': 3.74, 'I': 0.04712, 's_min': -0.4189, 's_max': 0.4189, 'sv_min': -3.2,
                #                'sv_max': 3.2, 'v_switch': 7.319, 'a_max': 9.51, 'v_min': -5.0, 'v_max': 20.0,
                #                'width': 0.31, 'length': 0.58}  F1/10 car
                self.params = {'mu': 1.0489, 'C_Sf': 20.898, 'C_Sr': 20.898, 'lf': 0.88392, 'lr': 1.50876, 'h': 0.59436,
                               'm': 1225.887, 'I': 1538.853371, 's_min': -0.910, 's_max': 0.910, 'sv_min': -0.4,
                               'sv_max': 0.4, 'v_switch': 7.319, 'a_max': 9.51, 'v_min': -5.0, 'v_max': 20.0,
                               'width': 1.674, 'length': 4.298}
            elif self.model == 'MB':
                self.params = {
                    # vehicle body dimensions
                    'length': 4.298,  # vehicle length [m]
                    'width': 1.674,  # vehicle width [m]

                    # steering constraints
                    's_min': -0.910,  # minimum steering angle [rad]
                    's_max': 0.910,  # maximum steering angle [rad]
                    'sv_min': -0.4,  # minimum steering velocity [rad/s]
                    'sv_max': 0.4,  # maximum steering velocity [rad/s]

                    # longitudinal constraints
                    'v_min': -13.9,  # minimum velocity [m/s]
                    'v_max': 45.8,  # minimum velocity [m/s]
                    'v_switch': 4.755,  # switching velocity [m/s]
                    'a_max': 11.5,  # maximum absolute acceleration [m/s^2]

                    # masses
                    'm': 1225.887,  # vehicle mass [kg]  MASS
                    'm_s': 1094.542,  # sprung mass [kg]  SMASS
                    'm_uf': 65.672,  # unsprung mass front [kg]  UMASSF
                    'm_ur': 65.672,  # unsprung mass rear [kg]  UMASSR

                    # axes distances
                    'lf': 0.88392,  # distance from spring mass center of gravity to front axle [m]  LENA
                    'lr': 1.50876,  # distance from spring mass center of gravity to rear axle [m]  LENB

                    # moments of inertia of sprung mass
                    'I_Phi_s': 244.0472306,  # moment of inertia for sprung mass in roll [kg m^2]  IXS
                    'I_y_s': 1342.259768,  # moment of inertia for sprung mass in pitch [kg m^2]  IYS
                    'I_z': 1538.853371,  # moment of inertia for sprung mass in yaw [kg m^2]  IZZ
                    'I_xz_s': 0.0,  # moment of inertia cross product [kg m^2]  IXZ

                    # suspension parameters
                    'K_sf': 21898.332429,  # suspension spring rate (front) [N/m]  KSF
                    'K_sdf': 1459.390293,  # suspension damping rate (front) [N s/m]  KSDF
                    'K_sr': 21898.332429,  # suspension spring rate (rear) [N/m]  KSR
                    'K_sdr': 1459.390293,  # suspension damping rate (rear) [N s/m]  KSDR

                    # geometric parameters
                    'T_f': 1.389888,  # track width front [m]  TRWF
                    'T_r': 1.423416,  # track width rear [m]  TRWB
                    'K_ras': 175186.659437,  # lateral spring rate at compliant compliant pin joint between M_s and M_u [N/m]  KRAS

                    'K_tsf': -12880.270509,  # auxiliary torsion roll stiffness per axle (normally negative) (front) [N m/rad]  KTSF
                    'K_tsr': 0.0,  # auxiliary torsion roll stiffness per axle (normally negative) (rear) [N m/rad]  KTSR
                    'K_rad': 10215.732056,  # damping rate at compliant compliant pin joint between M_s and M_u [N s/m]  KRADP
                    'K_zt': 189785.547723,  # vertical spring rate of tire [N/m]  TSPRINGR

                    'h_cg': 0.557784,  # center of gravity height of total mass [m]  HCG (mainly required for conversion to other vehicle models)
                    'h_raf': 0.0,  # height of roll axis above ground (front) [m]  HRAF
                    'h_rar': 0.0,  # height of roll axis above ground (rear) [m]  HRAR

                    'h_s': 0.59436,  # M_s center of gravity above ground [m]  HS

                    'I_uf': 32.539630,  # moment of inertia for unsprung mass about x-axis (front) [kg m^2]  IXUF
                    'I_ur': 32.539630,  # moment of inertia for unsprung mass about x-axis (rear) [kg m^2]  IXUR
                    'I_y_w': 1.7,  # wheel inertia, from internet forum for 235/65 R 17 [kg m^2]

                    'K_lt': 1.0278264878518764e-05,  # lateral compliance rate of tire, wheel, and suspension, per tire [m/N]  KLT
                    'R_w': 0.344,  # effective wheel/tire radius  chosen as tire rolling radius RR  taken from ADAMS documentation [m]

                    # split of brake and engine torque
                    'T_sb': 0.76,
                    'T_se': 1,

                    # suspension parameters
                    'D_f': -0.623359580, # [rad/m]  DF
                    'D_r': -0.209973753,  # [rad/m]  DR
                    'E_f': 0,  # [needs conversion if nonzero]  EF
                    'E_r': 0,  # [needs conversion if nonzero]  ER

                    # tire parameters from ADAMS handbook
                    # longitudinal coefficients
                    'tire_p_cx1': 1.6411,  # Shape factor Cfx for longitudinal force
                    'tire_p_dx1': 1.1739,  # Longitudinal friction Mux at Fznom
                    'tire_p_dx3': 0,  # Variation of friction Mux with camber
                    'tire_p_ex1': 0.46403,  # Longitudinal curvature Efx at Fznom
                    'tire_p_kx1': 22.303,  # Longitudinal slip stiffness Kfx/Fz at Fznom
                    'tire_p_hx1': 0.0012297,  # Horizontal shift Shx at Fznom
                    'tire_p_vx1': -8.8098e-006,  # Vertical shift Svx/Fz at Fznom
                    'tire_r_bx1': 13.276,  # Slope factor for combined slip Fx reduction
                    'tire_r_bx2': -13.778,  # Variation of slope Fx reduction with kappa
                    'tire_r_cx1': 1.2568,  # Shape factor for combined slip Fx reduction
                    'tire_r_ex1': 0.65225,  # Curvature factor of combined Fx
                    'tire_r_hx1': 0.0050722,  # Shift factor for combined slip Fx reduction

                    # lateral coefficients
                    'tire_p_cy1': 1.3507,  # Shape factor Cfy for lateral forces
                    'tire_p_dy1': 1.0489,  # Lateral friction Muy
                    'tire_p_dy3': -2.8821,  # Variation of friction Muy with squared camber
                    'tire_p_ey1': -0.0074722,  # Lateral curvature Efy at Fznom
                    'tire_p_ky1': -21.92,  # Maximum value of stiffness Kfy/Fznom
                    'tire_p_hy1': 0.0026747,  # Horizontal shift Shy at Fznom
                    'tire_p_hy3': 0.031415,  # Variation of shift Shy with camber
                    'tire_p_vy1': 0.037318,  # Vertical shift in Svy/Fz at Fznom
                    'tire_p_vy3': -0.32931,  # Variation of shift Svy/Fz with camber
                    'tire_r_by1': 7.1433,  # Slope factor for combined Fy reduction
                    'tire_r_by2': 9.1916,  # Variation of slope Fy reduction with alpha
                    'tire_r_by3': -0.027856,  # Shift term for alpha in slope Fy reduction
                    'tire_r_cy1': 1.0719,  # Shape factor for combined Fy reduction
                    'tire_r_ey1': -0.27572,  # Curvature factor of combined Fy
                    'tire_r_hy1': 5.7448e-006,  # Shift factor for combined Fy reduction
                    'tire_r_vy1': -0.027825,  # Kappa induced side force Svyk/Muy*Fz at Fznom
                    'tire_r_vy3': -0.27568,  # Variation of Svyk/Muy*Fz with camber
                    'tire_r_vy4': 12.12,  # Variation of Svyk/Muy*Fz with alpha
                    'tire_r_vy5': 1.9,  # Variation of Svyk/Muy*Fz with kappa
                    'tire_r_vy6': -10.704,  # Variation of Svyk/Muy*Fz with atan(kappa)
                }

        # simulation parameters
        try:
            self.num_agents = kwargs['num_agents']
        except:
            self.num_agents = 2

        try:
            self.drive_control_mode = kwargs['drive_control_mode']
        except:
            self.drive_control_mode = 'vel'
        # check valid options
        assert self.drive_control_mode in ['vel', 'acc']

        try:
            self.steering_control_mode = kwargs['steering_control_mode']
        except:
            self.steering_control_mode = 'angle'
        # check valid options
        assert self.steering_control_mode in ['angle', 'vel']

        try:
            self.timestep = kwargs['timestep']
        except:
            self.timestep = 0.01

        # default ego index
        try:
            self.ego_idx = kwargs['ego_idx']
        except:
            self.ego_idx = 0

        # radius to consider done
        self.start_thresh = 0.5  # 10cm

        # env states
        self.poses_x = []
        self.poses_y = []
        self.poses_theta = []
        self.collisions = np.zeros((self.num_agents, ))
        # TODO: collision_idx not used yet
        # self.collision_idx = -1 * np.ones((self.num_agents, ))

        # loop completion
        self.near_start = True
        self.num_toggles = 0

        # race info
        self.lap_times = np.zeros((self.num_agents, ))
        self.lap_counts = np.zeros((self.num_agents, ))
        self.current_time = 0.0

        # finish line info
        self.num_toggles = 0
        self.near_start = True
        self.near_starts = np.array([True]*self.num_agents)
        self.toggle_list = np.zeros((self.num_agents,))
        self.start_xs = np.zeros((self.num_agents, ))
        self.start_ys = np.zeros((self.num_agents, ))
        self.start_thetas = np.zeros((self.num_agents, ))
        self.start_rot = np.eye(2)

        # initiate stuff
        self.sim = Simulator(self.model, self.steering_control_mode, self.drive_control_mode, self.params,
                             self.num_agents, self.seed, time_step=self.timestep)
        self.sim.set_map(self.map_path, self.map_ext)

        # stateful observations for rendering
        self.render_obs = None

    def __del__(self):
        """
        Finalizer, does cleanup
        """
        pass

    def _check_done(self):
        """
        Check if the current rollout is done
        
        Args:
            None

        Returns:
            done (bool): whether the rollout is done
            toggle_list (list[int]): each agent's toggle list for crossing the finish zone
        """

        # this is assuming 2 agents
        # TODO: switch to maybe s-based
        left_t = 2
        right_t = 2
        
        poses_x = np.array(self.poses_x)-self.start_xs
        poses_y = np.array(self.poses_y)-self.start_ys
        delta_pt = np.dot(self.start_rot, np.stack((poses_x, poses_y), axis=0))
        temp_y = delta_pt[1,:]
        idx1 = temp_y > left_t
        idx2 = temp_y < -right_t
        temp_y[idx1] -= left_t
        temp_y[idx2] = -right_t - temp_y[idx2]
        temp_y[np.invert(np.logical_or(idx1, idx2))] = 0

        dist2 = delta_pt[0, :] ** 2 + temp_y ** 2
        closes = dist2 <= 15.0  # changed to work with 1:1 cars
        for i in range(self.num_agents):
            if closes[i] and not self.near_starts[i]:
                self.near_starts[i] = True
                self.toggle_list[i] += 1
            elif not closes[i] and self.near_starts[i]:
                self.near_starts[i] = False
                self.toggle_list[i] += 1
            self.lap_counts[i] = self.toggle_list[i] // 2
            if self.toggle_list[i] < 4:
                self.lap_times[i] = self.current_time
        
        done = (self.collisions[self.ego_idx]) or np.all(self.toggle_list >= 4)
        
        return done, self.toggle_list >= 4

    def _update_state(self, obs_dict):
        """
        Update the env's states according to observations
        
        Args:
            obs_dict (dict): dictionary of observation

        Returns:
            None
        """
        self.poses_x = obs_dict['poses_x']
        self.poses_y = obs_dict['poses_y']
        self.poses_theta = obs_dict['poses_theta']
        self.collisions = obs_dict['collisions']

    def step(self, action):
        """
        Step function for the gym env

        Args:
            action (np.ndarray(num_agents, 2))

        Returns:
            obs (dict): observation of the current step
            reward (float, default=self.timestep): step reward, currently is physics timestep
            done (bool): if the simulation is done
            info (dict): auxillary information dictionary
        """
        
        # call simulation step
        self.sim.step(action)
        obs, done, info = self.get_observations()

        # times
        reward = self.timestep
        self.current_time = self.current_time + self.timestep

        return obs, reward, done, info

    def reset(self, initial_states):
        """
        Reset the gym environment by given poses

        Args:
            initial_states (np.ndarray (num_agents, 7 or 3)): initial_states to reset agents to.
                            [x,y,yaw] or [x,y,yaw,steering angle, velocity, yaw_rate, beta]

        Returns:
            obs (dict): observation of the current step
            reward (float, default=self.timestep): step reward, currently is physics timestep
            done (bool): if the simulation is done
            info (dict): auxillary information dictionary
        """

        if initial_states.shape[1] == 3:  # to support legacy code
            temp_states = np.zeros((initial_states.shape[0], 7))
            temp_states[:, [0, 1, 2]] = initial_states  # keep first three states
            initial_states = temp_states  # fill rest with zeros

        # reset counters and data members
        self.current_time = 0.0
        self.collisions = np.zeros((self.num_agents, ))
        self.num_toggles = 0
        self.near_start = True
        self.near_starts = np.array([True]*self.num_agents)
        self.toggle_list = np.zeros((self.num_agents,))

        # states after reset
        self.start_xs = initial_states[:, 0]
        self.start_ys = initial_states[:, 1]
        self.start_thetas = initial_states[:, 2]
        self.start_rot = np.array(
            [[np.cos(-self.start_thetas[self.ego_idx]), -np.sin(-self.start_thetas[self.ego_idx])],
             [np.sin(-self.start_thetas[self.ego_idx]), np.cos(-self.start_thetas[self.ego_idx])]])

        # call reset to simulator
        self.sim.reset(initial_states)

        # get no input observations
        obs, done, info = self.get_observations()
        reward = 0

        self.render_obs = {
            'ego_idx': obs['ego_idx'],
            'poses_x': obs['poses_x'],
            'poses_y': obs['poses_y'],
            'poses_theta': obs['poses_theta'],
            'lap_times': obs['lap_times'],
            'lap_counts': obs['lap_counts']
            }
        
        return obs, reward, done, info

    def get_observations(self):
        # call simulation step
        # obs = self.sim.step(action)
        obs = self.sim.get_observations()
        obs['lap_times'] = self.lap_times
        obs['lap_counts'] = self.lap_counts

        F110Env.current_obs = obs

        self.render_obs = {
            'ego_idx': obs['ego_idx'],
            'poses_x': obs['poses_x'],
            'poses_y': obs['poses_y'],
            'poses_theta': obs['poses_theta'],
            'lap_times': obs['lap_times'],
            'lap_counts': obs['lap_counts']
        }

        # update data member
        self._update_state(obs)

        # check done
        done, toggle_list = self._check_done()
        info = {'checkpoint_done': toggle_list}

        return obs, done, info

    def update_map(self, map_path, map_ext):
        """
        Updates the map used by simulation

        Args:
            map_path (str): absolute path to the map yaml file
            map_ext (str): extension of the map image file

        Returns:
            None
        """
        self.sim.set_map(map_path, map_ext)

    def update_params(self, params, index=-1):
        """
        Updates the parameters used by simulation for vehicles
        
        Args:
            params (dict): dictionary of parameters
            index (int, default=-1): if >= 0 then only update a specific agent's params

        Returns:
            None
        """
        self.sim.update_params(params, agent_idx=index)

    def add_render_callback(self, callback_func):
        """
        Add extra drawing function to call during rendering.

        Args:
            callback_func (function (EnvRenderer) -> None): custom function to called during render()
        """

        F110Env.render_callbacks.append(callback_func)

    def render(self, mode='human'):
        """
        Renders the environment with pyglet. Use mouse scroll in the window to zoom in/out, use mouse click drag to pan. Shows the agents, the map, current fps (bottom left corner), and the race information near as text.

        Args:
            mode (str, default='human'): rendering mode, currently supports:
                'human': slowed down rendering such that the env is rendered in a way that sim time elapsed is close to real time elapsed
                'human_fast': render as fast as possible

        Returns:
            None
        """
        assert mode in ['human', 'human_fast']
        
        if F110Env.renderer is None:
            # first call, initialize everything
            from f110_gym.envs.rendering import EnvRenderer
            F110Env.renderer = EnvRenderer(WINDOW_W, WINDOW_H)
            F110Env.renderer.update_map(self.map_name, self.map_ext)
            
        F110Env.renderer.update_obs(self.render_obs)

        for render_callback in F110Env.render_callbacks:
            render_callback(F110Env.renderer)
        
        F110Env.renderer.dispatch_events()
        F110Env.renderer.on_draw()
        F110Env.renderer.flip()
        if mode == 'human':
            time.sleep(0.1)
        elif mode == 'human_fast':
            pass
