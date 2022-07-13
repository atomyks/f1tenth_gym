# Copyright 2020 Technical University of Munich, Professorship of Cyber-Physical Systems, Matthew O'Kelly, Aman Sinha, Hongrui Zheng

# Redistribution and use in source and binary forms, with or without modification, are permitted provided that the following conditions are met:

# 1. Redistributions of source code must retain the above copyright notice, this list of conditions and the following disclaimer.

# 2. Redistributions in binary form must reproduce the above copyright notice, this list of conditions and the following disclaimer in the documentation and/or other materials provided with the distribution.

# 3. Neither the name of the copyright holder nor the names of its contributors may be used to endorse or promote products derived from this software without specific prior written permission.

# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.



"""
Prototype of vehicle dynamics functions and classes for simulating 2D Single
Track dynamic model
Following the implementation of commanroad's Single Track Dynamics model
Original implementation: https://gitlab.lrz.de/tum-cps/commonroad-vehicle-models/
Author: Hongrui Zheng
"""

import numpy as np
from numba import njit

import unittest
import time
import f110_gym.envs.tire_models as tireModel

@njit(cache=True)
def accl_constraints(vel, accl, v_switch, a_max, v_min, v_max):
    """
    Acceleration constraints, adjusts the acceleration based on constraints

        Args:
            vel (float): current velocity of the vehicle
            accl (float): unconstraint desired acceleration
            v_switch (float): switching velocity (velocity at which the acceleration is no longer able to create wheel spin)
            a_max (float): maximum allowed acceleration
            v_min (float): minimum allowed velocity
            v_max (float): maximum allowed velocity

        Returns:
            accl (float): adjusted acceleration
    """

    # positive accl limit
    if vel > v_switch:
        pos_limit = a_max*v_switch/vel
    else:
        pos_limit = a_max

    # accl limit reached?
    if (vel <= v_min and accl <= 0) or (vel >= v_max and accl >= 0):
        accl = 0.
    elif accl <= -a_max:
        accl = -a_max
    elif accl >= pos_limit:
        accl = pos_limit

    return accl

@njit(cache=True)
def steering_constraint(steering_angle, steering_velocity, s_min, s_max, sv_min, sv_max):
    """
    Steering constraints, adjusts the steering velocity based on constraints

        Args:
            steering_angle (float): current steering_angle of the vehicle
            steering_velocity (float): unconstraint desired steering_velocity
            s_min (float): minimum steering angle
            s_max (float): maximum steering angle
            sv_min (float): minimum steering velocity
            sv_max (float): maximum steering velocity

        Returns:
            steering_velocity (float): adjusted steering velocity
    """

    # constraint steering velocity
    if (steering_angle <= s_min and steering_velocity <= 0) or (steering_angle >= s_max and steering_velocity >= 0):
        steering_velocity = 0.
    elif steering_velocity <= sv_min:
        steering_velocity = sv_min
    elif steering_velocity >= sv_max:
        steering_velocity = sv_max

    return steering_velocity


@njit(cache=True)
def vehicle_dynamics_ks(x, u_init, mu, C_Sf, C_Sr, lf, lr, h, m, I, s_min, s_max, sv_min, sv_max, v_switch, a_max, v_min, v_max):
    """
    Single Track Kinematic Vehicle Dynamics.

        Args:
            x (numpy.ndarray (3, )): vehicle state vector (x1, x2, x3, x4, x5)
                x1: x position in global coordinates
                x2: y position in global coordinates
                x3: steering angle of front wheels
                x4: velocity in x direction
                x5: yaw angle
            u (numpy.ndarray (2, )): control input vector (u1, u2)
                u1: steering angle velocity of front wheels
                u2: longitudinal acceleration

        Returns:
            f (numpy.ndarray): right hand side of differential equations
    """
    # wheelbase
    lwb = lf + lr

    # constraints
    u = np.array([steering_constraint(x[2], u_init[0], s_min, s_max, sv_min, sv_max), accl_constraints(x[3], u_init[1], v_switch, a_max, v_min, v_max)])

    # system dynamics
    f = np.array([x[3]*np.cos(x[4]),
         x[3]*np.sin(x[4]),
         u[0],
         u[1],
         x[3]/lwb*np.tan(x[2])])
    return f

@njit(cache=True)
def vehicle_dynamics_st(x, u_init, mu, C_Sf, C_Sr, lf, lr, h, m, I, s_min, s_max, sv_min, sv_max, v_switch, a_max, v_min, v_max):
    """
    Single Track Dynamic Vehicle Dynamics.

        Args:
            x (numpy.ndarray (3, )): vehicle state vector (x1, x2, x3, x4, x5, x6, x7)
                x1: x position in global coordinates
                x2: y position in global coordinates
                x3: steering angle of front wheels
                x4: velocity in x direction
                x5: yaw angle
                x6: yaw rate
                x7: slip angle at vehicle center
            u (numpy.ndarray (2, )): control input vector (u1, u2)
                u1: steering angle velocity of front wheels
                u2: longitudinal acceleration

        Returns:
            f (numpy.ndarray): right hand side of differential equations
    """

    # gravity constant m/s^2
    g = 9.81

    # constraints
    u = np.array([steering_constraint(x[2], u_init[0], s_min, s_max, sv_min, sv_max), accl_constraints(x[3], u_init[1], v_switch, a_max, v_min, v_max)])

    # switch to kinematic model for small velocities
    if abs(x[3]) < 0.5:
        # wheelbase
        lwb = lf + lr

        # system dynamics
        x_ks = x[0:5]
        f_ks = vehicle_dynamics_ks(x_ks, u, mu, C_Sf, C_Sr, lf, lr, h, m, I, s_min, s_max, sv_min, sv_max, v_switch, a_max, v_min, v_max)
        f = np.hstack((f_ks, np.array([u[1]/lwb*np.tan(x[2])+x[3]/(lwb*np.cos(x[2])**2)*u[0],
        0])))

    else:
        # system dynamics
        f = np.array([x[3]*np.cos(x[6] + x[4]),
            x[3]*np.sin(x[6] + x[4]),
            u[0],
            u[1],
            x[5],
            -mu*m/(x[3]*I*(lr+lf))*(lf**2*C_Sf*(g*lr-u[1]*h) + lr**2*C_Sr*(g*lf + u[1]*h))*x[5] \
                +mu*m/(I*(lr+lf))*(lr*C_Sr*(g*lf + u[1]*h) - lf*C_Sf*(g*lr - u[1]*h))*x[6] \
                +mu*m/(I*(lr+lf))*lf*C_Sf*(g*lr - u[1]*h)*x[2],
            (mu/(x[3]**2*(lr+lf))*(C_Sr*(g*lf + u[1]*h)*lr - C_Sf*(g*lr - u[1]*h)*lf)-1)*x[5] \
                -mu/(x[3]*(lr+lf))*(C_Sr*(g*lf + u[1]*h) + C_Sf*(g*lr-u[1]*h))*x[6] \
                +mu/(x[3]*(lr+lf))*(C_Sf*(g*lr-u[1]*h))*x[2]])

    return f


def init_mb(init_state, params):
    # init_MB - generates the initial state vector for the multi-body model
    #
    # Syntax:
    #     x0 = init_MB(init_state, p)
    #
    # Inputs:
    #     init_state - core initial states
    #     p - parameter vector
    #
    # Outputs:
    #     x0 - initial state vector
    #
    # Example:
    #
    # See also: ---

    # Author:       Matthias Althoff
    # Written:      11-January-2017
    # Last update:  ---
    # Last revision:---

    # Parameters
    # steering constraints
    s_min = params[2]  # minimum steering angle [rad]
    s_max = params[3]  # maximum steering angle [rad]
    # longitudinal constraints
    v_min = params[6]  # minimum velocity [m/s]
    v_max = params[7] # minimum velocity [m/s]
    # masses
    m_s = params[11] # sprung mass [kg]  SMASS
    m_uf = params[12] # unsprung mass front [kg]  UMASSF
    m_ur = params[13] # unsprung mass rear [kg]  UMASSR
    # axes distances
    lf = params[14] # distance from spring mass center of gravity to front axle [m]  LENA
    lr = params[15]  # distance from spring mass center of gravity to rear axle [m]  LENB
    # geometric parameters
    K_zt = params[30] # vertical spring rate of tire [N/m]  TSPRINGR
    R_w = params[39]  # effective wheel/tire radius  chosen as tire rolling radius RR  taken from ADAMS documentation [m]
    # create equivalent bicycle parameters
    g = 9.81  # [m/s^2]

    # obtain initial states from vector
    sx0 = init_state[0] # x-position in a global coordinate system
    sy0 = init_state[1] # y-position in a global coordinate system
    delta0 = init_state[2] # steering angle of front wheels
    vel0 = init_state[3] # speed of the car
    Psi0 = init_state[4] # yaw angle
    dotPsi0 = init_state[5] # yaw rate
    beta0 = init_state[6] # slip angle

    if delta0 > s_max:
        delta0 = s_max

    if delta0 < s_min:
        delta0 = s_min

    if vel0 > v_max:
        vel0 = v_max

    if vel0 < v_min:
        vel0 = v_min

    # auxiliary initial states
    F0_z_f = m_s * g * lr / ((lf + lr)) + m_uf * g
    F0_z_r = m_s * g * lf / ((lf + lr)) + m_ur * g

    # sprung mass states
    x0 = np.zeros((29,))  # init initial state vector
    x0[0] = sx0  # x-position in a global coordinate system
    x0[1] = sy0  # y-position in a global coordinate system
    x0[2] = delta0  # steering angle of front wheels
    x0[3] = np.cos(beta0) * vel0  # velocity in x-direction
    x0[4] = Psi0  # yaw angle
    x0[5] = dotPsi0  # yaw rate
    x0[6] = 0  # roll angle
    x0[7] = 0  # roll rate
    x0[8] = 0  # pitch angle
    x0[9] = 0  # pitch rate
    x0[10] = np.sin(beta0) * vel0  # velocity in y-direction
    x0[11] = 0  # z-position (zero height corresponds to steady state solution)
    x0[12] = 0  # velocity in z-direction

    # unsprung mass states (front)
    x0[13] = 0  # roll angle front
    x0[14] = 0  # roll rate front
    x0[15] = np.sin(beta0) * vel0 + lf * dotPsi0  # velocity in y-direction front
    x0[16] = (F0_z_f) / (2 * K_zt)  # z-position front
    x0[17] = 0  # velocity in z-direction front

    # unsprung mass states (rear)
    x0[18] = 0  # roll angle rear
    x0[19] = 0  # roll rate rear
    x0[20] = np.sin(beta0) * vel0 - lr * dotPsi0  # velocity in y-direction rear
    x0[21] = (F0_z_r) / (2 * K_zt)  # z-position rear
    x0[22] = 0  # velocity in z-direction rear

    # wheel states
    x0[23] = x0[3] / (R_w)  # left front wheel angular speed
    x0[24] = x0[3] / (R_w)  # right front wheel angular speed
    x0[25] = x0[3] / (R_w)  # left rear wheel angular speed
    x0[26] = x0[3] / (R_w)  # right rear wheel angular speed

    x0[27] = 0  # delta_y_f
    x0[28] = 0  # delta_y_r

    return x0


# @njit(cache=True)
def vehicle_dynamics_mb(x, u_init, params):
    """
    vehicleDynamics_mb - multi-body vehicle dynamics based on the DOT (department of transportation) vehicle dynamics
    reference point: center of mass

    Syntax:
        f = vehicleDynamics_mb(x,u,p)

    Inputs:
        :param x: vehicle state vector
        :param uInit: vehicle input vector
        :param params: vehicle parameter vector

    Outputs:
        :return f: right-hand side of differential equations

    Author: Matthias Althoff
    Written: 05-January-2017
    Last update: 17-December-2017
    Last revision: ---
    """

    #------------- BEGIN CODE --------------

    # set gravity constant
    g = 9.81  #[m/s^2]

    #states
    #x1 = x-position in a global coordinate system
    #x2 = y-position in a global coordinate system
    #x3 = steering angle of front wheels
    #x4 = velocity in x-direction
    #x5 = yaw angle
    #x6 = yaw rate

    #x7 = roll angle
    #x8 = roll rate
    #x9 = pitch angle
    #x10 = pitch rate
    #x11 = velocity in y-direction
    #x12 = z-position
    #x13 = velocity in z-direction

    #x14 = roll angle front
    #x15 = roll rate front
    #x16 = velocity in y-direction front
    #x17 = z-position front
    #x18 = velocity in z-direction front

    #x19 = roll angle rear
    #x20 = roll rate rear
    #x21 = velocity in y-direction rear
    #x22 = z-position rear
    #x23 = velocity in z-direction rear

    #x24 = left front wheel angular speed
    #x25 = right front wheel angular speed
    #x26 = left rear wheel angular speed
    #x27 = right rear wheel angular speed

    #x28 = delta_y_f
    #x29 = delta_y_r

    #u1 = steering angle velocity of front wheels
    #u2 = acceleration


    # vehicle body dimensions
    length =  params[0]  # vehicle length [m]
    width = params[1]  # vehicle width [m]

    # steering constraints
    s_min = params[2]  # minimum steering angle [rad]
    s_max = params[3]  # maximum steering angle [rad]
    sv_min = params[4] # minimum steering velocity [rad/s]
    sv_max = params[5] # maximum steering velocity [rad/s]

    # longitudinal constraints
    v_min = params[6]  # minimum velocity [m/s]
    v_max = params[7] # minimum velocity [m/s]
    v_switch = params[8]  # switching velocity [m/s]
    a_max = params[9] # maximum absolute acceleration [m/s^2]

    # masses
    m = params[10] # vehicle mass [kg]  MASS
    m_s = params[11] # sprung mass [kg]  SMASS
    m_uf = params[12] # unsprung mass front [kg]  UMASSF
    m_ur = params[13] # unsprung mass rear [kg]  UMASSR

    # axes distances
    lf = params[14] # distance from spring mass center of gravity to front axle [m]  LENA
    lr = params[15]  # distance from spring mass center of gravity to rear axle [m]  LENB

    # moments of inertia of sprung mass
    I_Phi_s = params[16]  # moment of inertia for sprung mass in roll [kg m^2]  IXS
    I_y_s = params[17]  # moment of inertia for sprung mass in pitch [kg m^2]  IYS
    I_z = params[18]  # moment of inertia for sprung mass in yaw [kg m^2]  IZZ
    I_xz_s = params[19]  # moment of inertia cross product [kg m^2]  IXZ

    # suspension parameters
    K_sf = params[20]  # suspension spring rate (front) [N/m]  KSF
    K_sdf = params[21]  # suspension damping rate (front) [N s/m]  KSDF
    K_sr = params[22]  # suspension spring rate (rear) [N/m]  KSR
    K_sdr = params[23]  # suspension damping rate (rear) [N s/m]  KSDR

    # geometric parameters
    T_f = params[24]   # track width front [m]  TRWF
    T_r = params[25]   # track width rear [m]  TRWB
    K_ras = params[26] # lateral spring rate at compliant compliant pin joint between M_s and M_u [N/m]  KRAS

    K_tsf = params[27]   # auxiliary torsion roll stiffness per axle (normally negative) (front) [N m/rad]  KTSF
    K_tsr = params[28] # auxiliary torsion roll stiffness per axle (normally negative) (rear) [N m/rad]  KTSR
    K_rad = params[29] # damping rate at compliant compliant pin joint between M_s and M_u [N s/m]  KRADP
    K_zt = params[30] # vertical spring rate of tire [N/m]  TSPRINGR

    h_cg = params[31]   # center of gravity height of total mass [m]  HCG (mainly required for conversion to other vehicle models)
    h_raf = params[32] # height of roll axis above ground (front) [m]  HRAF
    h_rar = params[33] # height of roll axis above ground (rear) [m]  HRAR

    h_s = params[34]   # M_s center of gravity above ground [m]  HS

    I_uf = params[35] # moment of inertia for unsprung mass about x-axis (front) [kg m^2]  IXUF
    I_ur = params[36] # moment of inertia for unsprung mass about x-axis (rear) [kg m^2]  IXUR
    I_y_w = params[37]  # wheel inertia, from internet forum for 235/65 R 17 [kg m^2]

    K_lt = params[38]  # lateral compliance rate of tire, wheel, and suspension, per tire [m/N]  KLT
    R_w = params[39]  # effective wheel/tire radius  chosen as tire rolling radius RR  taken from ADAMS documentation [m]

    # split of brake and engine torque
    T_sb = params[40]
    T_se = params[41]

    # suspension parameters
    D_f = params[42]  # [rad/m]  DF
    D_r = params[43]  # [rad/m]  DR
    E_f = params[44]  # [needs conversion if nonzero]  EF
    E_r = params[45]  # [needs conversion if nonzero]  ER


    # consider steering and acceleration constraints
    u = np.array([steering_constraint(x[2], u_init[0], s_min, s_max, sv_min, sv_max), accl_constraints(x[3], u_init[1], v_switch, a_max, v_min, v_max)])

    #compute slip angle at cg
    #switch to kinematic model for small velocities
    if abs(x[3]) < 0.1:
        beta = 0.
    else:
        beta = np.arctan(x[10]/x[3])
    vel = np.sqrt(x[3]**2 + x[10]**2)

    #vertical tire forces
    F_z_LF = (x[16] + R_w*(np.cos(x[13]) - 1) - 0.5*T_f*np.sin(x[13]))*K_zt
    F_z_RF = (x[16] + R_w*(np.cos(x[13]) - 1) + 0.5*T_f*np.sin(x[13]))*K_zt
    F_z_LR = (x[21] + R_w*(np.cos(x[18]) - 1) - 0.5*T_r*np.sin(x[18]))*K_zt
    F_z_RR = (x[21] + R_w*(np.cos(x[18]) - 1) + 0.5*T_r*np.sin(x[18]))*K_zt

    #obtain individual tire speeds
    u_w_lf = (x[3] + 0.5*T_f*x[5])*np.cos(x[2]) + (x[10] + lf*x[5])*np.sin(x[2])
    u_w_rf = (x[3] - 0.5*T_f*x[5])*np.cos(x[2]) + (x[10] + lf*x[5])*np.sin(x[2])
    u_w_lr = x[3] + 0.5*T_r*x[5]
    u_w_rr = x[3] - 0.5*T_r*x[5]

    #negative wheel spin forbidden
    if u_w_lf < 0.0:
        u_w_lf *= 0

    if u_w_rf < 0.0:
        u_w_rf *= 0

    if u_w_lr < 0.0:
        u_w_lr *= 0

    if u_w_rr < 0.0:
        u_w_rr *= 0
    #compute longitudinal slip
    #switch to kinematic model for small velocities
    if abs(x[3]) < 0.1:
        s_lf = 0.
        s_rf = 0.
        s_lr = 0.
        s_rr = 0.
    else:
        s_lf = 1 - R_w*x[23]/u_w_lf
        s_rf = 1 - R_w*x[24]/u_w_rf
        s_lr = 1 - R_w*x[25]/u_w_lr
        s_rr = 1 - R_w*x[26]/u_w_rr

    #lateral slip angles
    #switch to kinematic model for small velocities
    if abs(x[3]) < 0.1:
        alpha_LF = 0.
        alpha_RF = 0.
        alpha_LR = 0.
        alpha_RR = 0.
    else:
        alpha_LF = np.arctan((x[10] + lf*x[5] - x[14]*(R_w - x[16]))/(x[3] + 0.5*T_f*x[5])) - x[2]
        alpha_RF = np.arctan((x[10] + lf*x[5] - x[14]*(R_w - x[16]))/(x[3] - 0.5*T_f*x[5])) - x[2]
        alpha_LR = np.arctan((x[10] - lr*x[5] - x[19]*(R_w - x[21]))/(x[3] + 0.5*T_r*x[5]))
        alpha_RR = np.arctan((x[10] - lr*x[5] - x[19]*(R_w - x[21]))/(x[3] - 0.5*T_r*x[5]))

    #auxiliary suspension movement
    z_SLF = (h_s - R_w + x[16] - x[11])/np.cos(x[6]) - h_s + R_w + lf*x[8] + 0.5*(x[6] - x[13])*T_f
    z_SRF = (h_s - R_w + x[16] - x[11])/np.cos(x[6]) - h_s + R_w + lf*x[8] - 0.5*(x[6] - x[13])*T_f
    z_SLR = (h_s - R_w + x[21] - x[11])/np.cos(x[6]) - h_s + R_w - lr*x[8] + 0.5*(x[6] - x[18])*T_r
    z_SRR = (h_s - R_w + x[21] - x[11])/np.cos(x[6]) - h_s + R_w - lr*x[8] - 0.5*(x[6] - x[18])*T_r

    dz_SLF = x[17] - x[12] + lf*x[9] + 0.5*(x[7] - x[14])*T_f
    dz_SRF = x[17] - x[12] + lf*x[9] - 0.5*(x[7] - x[14])*T_f
    dz_SLR = x[22] - x[12] - lr*x[9] + 0.5*(x[7] - x[19])*T_r
    dz_SRR = x[22] - x[12] - lr*x[9] - 0.5*(x[7] - x[19])*T_r

    #camber angles
    gamma_LF = x[6] + D_f*z_SLF + E_f*(z_SLF)**2
    gamma_RF = x[6] - D_f*z_SRF - E_f*(z_SRF)**2
    gamma_LR = x[6] + D_r*z_SLR + E_r*(z_SLR)**2
    gamma_RR = x[6] - D_r*z_SRR - E_r*(z_SRR)**2

    #compute longitudinal tire forces using the magic formula for pure slip
    F0_x_LF = tireModel.formula_longitudinal(s_lf, gamma_LF, F_z_LF, params) # TODO tire model
    F0_x_RF = tireModel.formula_longitudinal(s_rf, gamma_RF, F_z_RF, params)
    F0_x_LR = tireModel.formula_longitudinal(s_lr, gamma_LR, F_z_LR, params)
    F0_x_RR = tireModel.formula_longitudinal(s_rr, gamma_RR, F_z_RR, params)

    #compute lateral tire forces using the magic formula for pure slip
    res = tireModel.formula_lateral(alpha_LF, gamma_LF, F_z_LF, params)
    F0_y_LF = res[0]
    mu_y_LF = res[1]
    res = tireModel.formula_lateral(alpha_RF, gamma_RF, F_z_RF, params)
    F0_y_RF = res[0]
    mu_y_RF = res[1]
    res = tireModel.formula_lateral(alpha_LR, gamma_LR, F_z_LR, params)
    F0_y_LR = res[0]
    mu_y_LR = res[1]
    res = tireModel.formula_lateral(alpha_RR, gamma_RR, F_z_RR, params)
    F0_y_RR = res[0]
    mu_y_RR = res[1]

    #compute longitudinal tire forces using the magic formula for combined slip
    F_x_LF = tireModel.formula_longitudinal_comb(s_lf, alpha_LF, F0_x_LF, params)
    F_x_RF = tireModel.formula_longitudinal_comb(s_rf, alpha_RF, F0_x_RF, params)
    F_x_LR = tireModel.formula_longitudinal_comb(s_lr, alpha_LR, F0_x_LR, params)
    F_x_RR = tireModel.formula_longitudinal_comb(s_rr, alpha_RR, F0_x_RR, params)

    #compute lateral tire forces using the magic formula for combined slip
    F_y_LF = tireModel.formula_lateral_comb(s_lf, alpha_LF, gamma_LF, mu_y_LF, F_z_LF, F0_y_LF, params)
    F_y_RF = tireModel.formula_lateral_comb(s_rf, alpha_RF, gamma_RF, mu_y_RF, F_z_RF, F0_y_RF, params)
    F_y_LR = tireModel.formula_lateral_comb(s_lr, alpha_LR, gamma_LR, mu_y_LR, F_z_LR, F0_y_LR, params)
    F_y_RR = tireModel.formula_lateral_comb(s_rr, alpha_RR, gamma_RR, mu_y_RR, F_z_RR, F0_y_RR, params)

    #auxiliary movements for compliant joint equations
    delta_z_f = h_s - R_w + x[16] - x[11]
    delta_z_r = h_s - R_w + x[21] - x[11]

    delta_phi_f = x[6] - x[13]
    delta_phi_r = x[6] - x[18]

    dot_delta_phi_f = x[7] - x[14]
    dot_delta_phi_r = x[7] - x[19]

    dot_delta_z_f = x[17] - x[12]
    dot_delta_z_r = x[22] - x[12]

    dot_delta_y_f = x[10] + lf*x[5] - x[15]
    dot_delta_y_r = x[10] - lr*x[5] - x[20]

    delta_f = delta_z_f*np.sin(x[6]) - x[27]*np.cos(x[6]) - (h_raf - R_w)*np.sin(delta_phi_f)
    delta_r = delta_z_r*np.sin(x[6]) - x[28]*np.cos(x[6]) - (h_rar - R_w)*np.sin(delta_phi_r)

    dot_delta_f = (delta_z_f*np.cos(x[6]) + x[27]*np.sin(x[6]))*x[7] + dot_delta_z_f*np.sin(x[6]) - dot_delta_y_f*np.cos(x[6]) - (h_raf - R_w)*np.cos(delta_phi_f)*dot_delta_phi_f
    dot_delta_r = (delta_z_r*np.cos(x[6]) + x[28]*np.sin(x[6]))*x[7] + dot_delta_z_r*np.sin(x[6]) - dot_delta_y_r*np.cos(x[6]) - (h_rar - R_w)*np.cos(delta_phi_r)*dot_delta_phi_r

    #compliant joint forces
    F_RAF = delta_f*K_ras + dot_delta_f*K_rad
    F_RAR = delta_r*K_ras + dot_delta_r*K_rad

    #auxiliary suspension forces (bump stop neglected  squat/lift forces neglected)
    F_SLF = m_s*g*lr/(2*(lf+lr)) - z_SLF*K_sf - dz_SLF*K_sdf + (x[6] - x[13])*K_tsf/T_f

    F_SRF = m_s*g*lr/(2*(lf+lr)) - z_SRF*K_sf - dz_SRF*K_sdf - (x[6] - x[13])*K_tsf/T_f

    F_SLR = m_s*g*lf/(2*(lf+lr)) - z_SLR*K_sr - dz_SLR*K_sdr + (x[6] - x[18])*K_tsr/T_r

    F_SRR = m_s*g*lf/(2*(lf+lr)) - z_SRR*K_sr - dz_SRR*K_sdr - (x[6] - x[18])*K_tsr/T_r


    #auxiliary variables sprung mass
    sumX = F_x_LR + F_x_RR + (F_x_LF + F_x_RF)*np.cos(x[2]) - (F_y_LF + F_y_RF)*np.sin(x[2])

    sumN = (F_y_LF + F_y_RF)*lf*np.cos(x[2]) + (F_x_LF + F_x_RF)*lf*np.sin(x[2]) \
           + (F_y_RF - F_y_LF)*0.5*T_f*np.sin(x[2]) + (F_x_LF - F_x_RF)*0.5*T_f*np.cos(x[2]) \
           + (F_x_LR - F_x_RR)*0.5*T_r - (F_y_LR + F_y_RR)*lr

    sumY_s = (F_RAF + F_RAR)*np.cos(x[6]) + (F_SLF + F_SLR + F_SRF + F_SRR)*np.sin(x[6])

    sumL = 0.5*F_SLF*T_f + 0.5*F_SLR*T_r - 0.5*F_SRF*T_f - 0.5*F_SRR*T_r \
           - F_RAF/np.cos(x[6])*(h_s - x[11] - R_w + x[16] - (h_raf - R_w)*np.cos(x[13])) \
           - F_RAR/np.cos(x[6])*(h_s - x[11] - R_w + x[21] - (h_rar - R_w)*np.cos(x[18]))

    sumZ_s = (F_SLF + F_SLR + F_SRF + F_SRR)*np.cos(x[6]) - (F_RAF + F_RAR)*np.sin(x[6])

    sumM_s = lf*(F_SLF + F_SRF) - lr*(F_SLR + F_SRR) + ((F_x_LF + F_x_RF)*np.cos(x[2]) \
                                                          - (F_y_LF + F_y_RF)*np.sin(x[2]) + F_x_LR + F_x_RR)*(h_s - x[11])

    #auxiliary variables unsprung mass
    sumL_uf = 0.5*F_SRF*T_f - 0.5*F_SLF*T_f - F_RAF*(h_raf - R_w) \
              + F_z_LF*(R_w*np.sin(x[13]) + 0.5*T_f*np.cos(x[13]) - K_lt*F_y_LF) \
              - F_z_RF*(-R_w*np.sin(x[13]) + 0.5*T_f*np.cos(x[13]) + K_lt*F_y_RF) \
              - ((F_y_LF + F_y_RF)*np.cos(x[2]) + (F_x_LF + F_x_RF)*np.sin(x[2]))*(R_w - x[16])

    sumL_ur = 0.5*F_SRR*T_r - 0.5*F_SLR*T_r - F_RAR*(h_rar - R_w) \
              + F_z_LR*(R_w*np.sin(x[18]) + 0.5*T_r*np.cos(x[18]) - K_lt*F_y_LR) \
              - F_z_RR*(-R_w*np.sin(x[18]) + 0.5*T_r*np.cos(x[18]) + K_lt*F_y_RR) \
              - (F_y_LR + F_y_RR)*(R_w - x[21])

    sumZ_uf = F_z_LF + F_z_RF + F_RAF*np.sin(x[6]) - (F_SLF + F_SRF)*np.cos(x[6])

    sumZ_ur = F_z_LR + F_z_RR + F_RAR*np.sin(x[6]) - (F_SLR + F_SRR)*np.cos(x[6])

    sumY_uf = (F_y_LF + F_y_RF)*np.cos(x[2]) + (F_x_LF + F_x_RF)*np.sin(x[2]) \
              - F_RAF*np.cos(x[6]) - (F_SLF + F_SRF)*np.sin(x[6])

    sumY_ur = (F_y_LR + F_y_RR) \
              - F_RAR*np.cos(x[6]) - (F_SLR + F_SRR)*np.sin(x[6])


    # dynamics common with single-track model
    f = np.zeros((29,))# init 'right hand side'
    # switch to kinematic model for small velocities
    if abs(x[3]) < 0.1:
        # wheelbase
        # lwb = lf + lr

        # system dynamics
        # x_ks = [x[0],  x[1],  x[2],  x[3],  x[4]]
        # f_ks = vehicle_dynamics_ks(x_ks, u, p)
        # f.extend(f_ks)
        # f.append(u[1]*lwb*np.tan(x[2]) + x[3]/(lwb*np.cos(x[2])**2)*u[0])

        # Use kinematic model with reference point at center of mass
        # wheelbase
        lwb = lf + lr

        # system dynamics
        x_ks = x[0:5]
        # kinematic model
        # f_ks = vehicle_dynamics_ks_cog(x_ks, u, p)
        # f = [f_ks[0], f_ks[1], f_ks[2], f_ks[3], f_ks[4]] # 1 2 3 4 5
        f_ks = vehicle_dynamics_ks(x_ks, u, 0, 0, 0, lf, lr, 0, 0, 0, s_min, s_max, sv_min, sv_max, v_switch, a_max, v_min, v_max)
        f[0:5] = f_ks
        # derivative of slip angle and yaw rate
        d_beta = (lr * u[0]) / (lwb * np.cos(x[2]) ** 2 * (1 + (np.tan(x[2]) ** 2 * lr / lwb) ** 2))
        dd_psi = 1 / lwb * (u[1] * np.cos(x[6]) * np.tan(x[2]) -
                            x[3] * np.sin(x[6]) * d_beta * np.tan(x[2]) +
                            x[3] * np.cos(x[6]) * u[0] / np.cos(x[2]) ** 2)
        # f.append(dd_psi) # 6
        f[5] = dd_psi

    else:
        f[0] = np.cos(beta + x[4])*vel # 1
        f[1] = np.sin(beta + x[4])*vel # 2
        f[2] = u[0] # 3
        f[3] = 1/m*sumX + x[5]*x[10] # 4
        f[4] = x[5] # 5
        f[5] = 1/(I_z - (I_xz_s)**2/I_Phi_s)*(sumN + I_xz_s/I_Phi_s*sumL) # 6


    # remaining sprung mass dynamics
    f[6] = x[7] # 7
    f[7] = 1/(I_Phi_s - (I_xz_s)**2/I_z)*(I_xz_s/I_z*sumN + sumL) # 8
    f[8] = x[9] # 9
    f[9] = 1/I_y_s*sumM_s # 10
    f[10] = 1/m_s*sumY_s - x[5]*x[3] # 11
    f[11] = x[12] # 12
    f[12] = g - 1/m_s*sumZ_s # 13

    #unsprung mass dynamics (front)
    f[13] = x[14] # 14
    f[14] = 1/I_uf*sumL_uf # 15
    f[15] = 1/m_uf*sumY_uf - x[5]*x[3] # 16
    f[16] = x[17] # 17
    f[17] = g - 1/m_uf*sumZ_uf # 18

    #unsprung mass dynamics (rear)
    f[18] = x[19] # 19
    f[19] = 1/I_ur*sumL_ur # 20
    f[20] = 1/m_ur*sumY_ur - x[5]*x[3] # 21
    f[21] = x[22] # 22
    f[22] = g - 1/m_ur*sumZ_ur # 23

    #convert acceleration input to brake and engine torque
    if u[1]>0:
        T_B = 0.0
        T_E = m*R_w*u[1]
    else:
        T_B = m*R_w*u[1]
        T_E = 0.0



    #wheel dynamics (p.T  new parameter for torque splitting)
    f[23] = 1/I_y_w*(-R_w*F_x_LF + 0.5*T_sb*T_B + 0.5*T_se*T_E) # 24
    f[24] = 1/I_y_w*(-R_w*F_x_RF + 0.5*T_sb*T_B + 0.5*T_se*T_E) # 25
    f[25] = 1/I_y_w*(-R_w*F_x_LR + 0.5*(1-T_sb)*T_B + 0.5*(1-T_se)*T_E) # 26
    f[26] = 1/I_y_w*(-R_w*F_x_RR + 0.5*(1-T_sb)*T_B + 0.5*(1-T_se)*T_E) # 27

    #negative wheel spin forbidden
    for iState in range(23, 27):
        if x[iState] < 0.0:
            x[iState] = 0.0
            f[iState] = 0.0

    #compliant joint equations
    f[27] = dot_delta_y_f # 28
    f[28] = dot_delta_y_r # 29

    return f


@njit(cache=True)
def pid(speed, steer, current_speed, current_steer, max_sv, max_a, max_v, min_v):
    """
    Basic controller for speed/steer -> accl./steer vel.

        Args:
            speed (float): desired input speed
            steer (float): desired input steering angle

        Returns:
            accl (float): desired input acceleration
            sv (float): desired input steering velocity
    """
    # steering
    steer_diff = steer - current_steer
    if np.fabs(steer_diff) > 1e-4:
        sv = (steer_diff / np.fabs(steer_diff)) * max_sv
    else:
        sv = 0.0

    # accl
    vel_diff = speed - current_speed
    # currently forward
    if current_speed > 0.:
        if (vel_diff > 0):
            # accelerate
            kp = 10.0 * max_a / max_v
            accl = kp * vel_diff
        else:
            # braking
            kp = 10.0 * max_a / (-min_v)
            accl = kp * vel_diff
    # currently backwards
    else:
        if (vel_diff > 0):
            # braking
            kp = 2.0 * max_a / max_v
            accl = kp * vel_diff
        else:
            # accelerating
            kp = 2.0 * max_a / (-min_v)
            accl = kp * vel_diff

    return accl, sv

def func_KS(x, t, u, mu, C_Sf, C_Sr, lf, lr, h, m, I, s_min, s_max, sv_min, sv_max, v_switch, a_max, v_min, v_max):
    f = vehicle_dynamics_ks(x, u, mu, C_Sf, C_Sr, lf, lr, h, m, I, s_min, s_max, sv_min, sv_max, v_switch, a_max, v_min, v_max)
    return f

def func_ST(x, t, u, mu, C_Sf, C_Sr, lf, lr, h, m, I, s_min, s_max, sv_min, sv_max, v_switch, a_max, v_min, v_max):
    f = vehicle_dynamics_st(x, u, mu, C_Sf, C_Sr, lf, lr, h, m, I, s_min, s_max, sv_min, sv_max, v_switch, a_max, v_min, v_max)
    return f

class DynamicsTest(unittest.TestCase):
    def setUp(self):
        # test params
        self.mu = 1.0489
        self.C_Sf = 21.92/1.0489
        self.C_Sr = 21.92/1.0489
        self.lf = 0.3048*3.793293
        self.lr = 0.3048*4.667707
        self.h = 0.3048*2.01355
        self.m = 4.4482216152605/0.3048*74.91452
        self.I = 4.4482216152605*0.3048*1321.416

        #steering constraints
        self.s_min = -1.066  #minimum steering angle [rad]
        self.s_max = 1.066  #maximum steering angle [rad]
        self.sv_min = -0.4  #minimum steering velocity [rad/s]
        self.sv_max = 0.4  #maximum steering velocity [rad/s]

        #longitudinal constraints
        self.v_min = -13.6  #minimum velocity [m/s]
        self.v_max = 50.8  #minimum velocity [m/s]
        self.v_switch = 7.319  #switching velocity [m/s]
        self.a_max = 11.5  #maximum absolute acceleration [m/s^2]

    def test_derivatives(self):
        # ground truth derivatives
        f_ks_gt = [16.3475935934250209, 0.4819314886013121, 0.1500000000000000, 5.1464424102339752, 0.2401426578627629]
        f_st_gt = [15.7213512030862397, 0.0925527979719355, 0.1500000000000000, 5.3536773276413925, 0.0529001056654038, 0.6435589397748606, 0.0313297971641291]

        # system dynamics
        g = 9.81
        x_ks = np.array([3.9579422297936526, 0.0391650102771405, 0.0378491427211811, 16.3546957860883566, 0.0294717351052816])
        x_st = np.array([2.0233348142065677, 0.0041907137716636, 0.0197545248559617, 15.7216236334290116, 0.0025857914776859, 0.0529001056654038, 0.0033012170610298])
        v_delta = 0.15
        acc = 0.63*g
        u = np.array([v_delta,  acc])

        f_ks = vehicle_dynamics_ks(x_ks, u, self.mu, self.C_Sf, self.C_Sr, self.lf, self.lr, self.h, self.m, self.I, self.s_min, self.s_max, self.sv_min, self.sv_max, self.v_switch, self.a_max, self.v_min, self.v_max)
        f_st = vehicle_dynamics_st(x_st, u, self.mu, self.C_Sf, self.C_Sr, self.lf, self.lr, self.h, self.m, self.I, self.s_min, self.s_max, self.sv_min, self.sv_max, self.v_switch, self.a_max, self.v_min, self.v_max)

        start = time.time()
        for i in range(10000):
            f_st = vehicle_dynamics_st(x_st, u, self.mu, self.C_Sf, self.C_Sr, self.lf, self.lr, self.h, self.m, self.I, self.s_min, self.s_max, self.sv_min, self.sv_max, self.v_switch, self.a_max, self.v_min, self.v_max)
        duration = time.time() - start
        avg_fps = 10000/duration

        self.assertAlmostEqual(np.max(np.abs(f_ks_gt-f_ks)), 0.)
        self.assertAlmostEqual(np.max(np.abs(f_st_gt-f_st)), 0.)
        self.assertGreater(avg_fps, 5000)

    def test_zeroinit_roll(self):
        from scipy.integrate import odeint

        # testing for zero initial state, zero input singularities
        g = 9.81
        t_start = 0.
        t_final = 1.
        delta0 = 0.
        vel0 = 0.
        Psi0 = 0.
        dotPsi0 = 0.
        beta0 = 0.
        sy0 = 0.
        initial_state = [0,sy0,delta0,vel0,Psi0,dotPsi0,beta0]

        x0_KS = np.array(initial_state[0:5])
        x0_ST = np.array(initial_state)

        # time vector
        t = np.arange(t_start, t_final, 1e-4)

        # set input: rolling car (velocity should stay constant)
        u = np.array([0., 0.])

        # simulate single-track model
        x_roll_st = odeint(func_ST, x0_ST, t, args=(u, self.mu, self.C_Sf, self.C_Sr, self.lf, self.lr, self.h, self.m, self.I, self.s_min, self.s_max, self.sv_min, self.sv_max, self.v_switch, self.a_max, self.v_min, self.v_max))
        # simulate kinematic single-track model
        x_roll_ks = odeint(func_KS, x0_KS, t, args=(u, self.mu, self.C_Sf, self.C_Sr, self.lf, self.lr, self.h, self.m, self.I, self.s_min, self.s_max, self.sv_min, self.sv_max, self.v_switch, self.a_max, self.v_min, self.v_max))

        self.assertTrue(all(x_roll_st[-1]==x0_ST))
        self.assertTrue(all(x_roll_ks[-1]==x0_KS))

    def test_zeroinit_dec(self):
        from scipy.integrate import odeint

        # testing for zero initial state, decelerating input singularities
        g = 9.81
        t_start = 0.
        t_final = 1.
        delta0 = 0.
        vel0 = 0.
        Psi0 = 0.
        dotPsi0 = 0.
        beta0 = 0.
        sy0 = 0.
        initial_state = [0,sy0,delta0,vel0,Psi0,dotPsi0,beta0]

        x0_KS = np.array(initial_state[0:5])
        x0_ST = np.array(initial_state)

        # time vector
        t = np.arange(t_start, t_final, 1e-4)

        # set decel input
        u = np.array([0., -0.7*g])

        # simulate single-track model
        x_dec_st = odeint(func_ST, x0_ST, t, args=(u, self.mu, self.C_Sf, self.C_Sr, self.lf, self.lr, self.h, self.m, self.I, self.s_min, self.s_max, self.sv_min, self.sv_max, self.v_switch, self.a_max, self.v_min, self.v_max))
        # simulate kinematic single-track model
        x_dec_ks = odeint(func_KS, x0_KS, t, args=(u, self.mu, self.C_Sf, self.C_Sr, self.lf, self.lr, self.h, self.m, self.I, self.s_min, self.s_max, self.sv_min, self.sv_max, self.v_switch, self.a_max, self.v_min, self.v_max))

        # ground truth for single-track model
        x_dec_st_gt = [-3.4335000000000013, 0.0000000000000000, 0.0000000000000000, -6.8670000000000018, 0.0000000000000000, 0.0000000000000000, 0.0000000000000000]
        # ground truth for kinematic single-track model
        x_dec_ks_gt = [-3.4335000000000013, 0.0000000000000000, 0.0000000000000000, -6.8670000000000018, 0.0000000000000000]

        self.assertTrue(all(abs(x_dec_st[-1] - x_dec_st_gt) < 1e-2))
        self.assertTrue(all(abs(x_dec_ks[-1] - x_dec_ks_gt) < 1e-2))

    def test_zeroinit_acc(self):
        from scipy.integrate import odeint

        # testing for zero initial state, accelerating with left steer input singularities
        # wheel spin and velocity should increase more wheel spin at rear
        g = 9.81
        t_start = 0.
        t_final = 1.
        delta0 = 0.
        vel0 = 0.
        Psi0 = 0.
        dotPsi0 = 0.
        beta0 = 0.
        sy0 = 0.
        initial_state = [0,sy0,delta0,vel0,Psi0,dotPsi0,beta0]

        x0_KS = np.array(initial_state[0:5])
        x0_ST = np.array(initial_state)

        # time vector
        t = np.arange(t_start, t_final, 1e-4)

        # set decel input
        u = np.array([0.15, 0.63*g])

        # simulate single-track model
        x_acc_st = odeint(func_ST, x0_ST, t, args=(u, self.mu, self.C_Sf, self.C_Sr, self.lf, self.lr, self.h, self.m, self.I, self.s_min, self.s_max, self.sv_min, self.sv_max, self.v_switch, self.a_max, self.v_min, self.v_max))
        # simulate kinematic single-track model
        x_acc_ks = odeint(func_KS, x0_KS, t, args=(u, self.mu, self.C_Sf, self.C_Sr, self.lf, self.lr, self.h, self.m, self.I, self.s_min, self.s_max, self.sv_min, self.sv_max, self.v_switch, self.a_max, self.v_min, self.v_max))

        # ground truth for single-track model
        x_acc_st_gt = [3.0731976046859715, 0.2869835398304389, 0.1500000000000000, 6.1802999999999999, 0.1097747074946325, 0.3248268063223301, 0.0697547542798040]
        # ground truth for kinematic single-track model
        x_acc_ks_gt = [3.0845676868494927, 0.1484249221523042, 0.1500000000000000, 6.1803000000000017, 0.1203664469224163]

        self.assertTrue(all(abs(x_acc_st[-1] - x_acc_st_gt) < 1e-2))
        self.assertTrue(all(abs(x_acc_ks[-1] - x_acc_ks_gt) < 1e-2))

    def test_zeroinit_rollleft(self):
        from scipy.integrate import odeint

        # testing for zero initial state, rolling and steering left input singularities
        g = 9.81
        t_start = 0.
        t_final = 1.
        delta0 = 0.
        vel0 = 0.
        Psi0 = 0.
        dotPsi0 = 0.
        beta0 = 0.
        sy0 = 0.
        initial_state = [0,sy0,delta0,vel0,Psi0,dotPsi0,beta0]

        x0_KS = np.array(initial_state[0:5])
        x0_ST = np.array(initial_state)

        # time vector
        t = np.arange(t_start, t_final, 1e-4)

        # set decel input
        u = np.array([0.15, 0.])

        # simulate single-track model
        x_left_st = odeint(func_ST, x0_ST, t, args=(u, self.mu, self.C_Sf, self.C_Sr, self.lf, self.lr, self.h, self.m, self.I, self.s_min, self.s_max, self.sv_min, self.sv_max, self.v_switch, self.a_max, self.v_min, self.v_max))
        # simulate kinematic single-track model
        x_left_ks = odeint(func_KS, x0_KS, t, args=(u, self.mu, self.C_Sf, self.C_Sr, self.lf, self.lr, self.h, self.m, self.I, self.s_min, self.s_max, self.sv_min, self.sv_max, self.v_switch, self.a_max, self.v_min, self.v_max))

        # ground truth for single-track model
        x_left_st_gt = [0.0000000000000000, 0.0000000000000000, 0.1500000000000000, 0.0000000000000000, 0.0000000000000000, 0.0000000000000000, 0.0000000000000000]
        # ground truth for kinematic single-track model
        x_left_ks_gt = [0.0000000000000000, 0.0000000000000000, 0.1500000000000000, 0.0000000000000000, 0.0000000000000000]

        self.assertTrue(all(abs(x_left_st[-1] - x_left_st_gt) < 1e-2))
        self.assertTrue(all(abs(x_left_ks[-1] - x_left_ks_gt) < 1e-2))

if __name__ == '__main__':
    unittest.main()