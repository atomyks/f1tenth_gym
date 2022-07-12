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

@njit(cache=True)
def vehicle_dynamics_mb(x, uInit, params):
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

    # tire parameters from ADAMS handbook
    # longitudinal coefficients
    tire_p_cx1 = params[46]  # Shape factor Cfx for longitudinal force
    tire_p_dx1 = params[47]  # Longitudinal friction Mux at Fznom
    tire_p_dx3 = params[48]  # Variation of friction Mux with camber
    tire_p_ex1 = params[49]  # Longitudinal curvature Efx at Fznom
    tire_p_kx1 = params[50]  # Longitudinal slip stiffness Kfx/Fz at Fznom
    tire_p_hx1 = params[51]  # Horizontal shift Shx at Fznom
    tire_p_vx1 = params[52]  # Vertical shift Svx/Fz at Fznom
    tire_r_bx1 = params[53]  # Slope factor for combined slip Fx reduction
    tire_r_bx2 = params[54]  # Variation of slope Fx reduction with kappa
    tire_r_cx1 = params[55]  # Shape factor for combined slip Fx reduction
    tire_r_ex1 = params[56]  # Curvature factor of combined Fx
    tire_r_hx1 = params[57]  # Shift factor for combined slip Fx reduction

    # lateral coefficients
    tire_p_cy1 = params[58]  # Shape factor Cfy for lateral forces
    tire_p_dy1 = params[59]  # Lateral friction Muy
    tire_p_dy3 = params[60]  # Variation of friction Muy with squared camber
    tire_p_ey1 = params[61]  # Lateral curvature Efy at Fznom
    tire_p_ky1 = params[62]  # Maximum value of stiffness Kfy/Fznom
    tire_p_hy1 = params[63]  # Horizontal shift Shy at Fznom
    tire_p_hy3 = params[64]  # Variation of shift Shy with camber
    tire_p_vy1 = params[65]  # Vertical shift in Svy/Fz at Fznom
    tire_p_vy3 = params[66]  # Variation of shift Svy/Fz with camber
    tire_r_by1 = params[67]  # Slope factor for combined Fy reduction
    tire_r_by2 = params[68]  # Variation of slope Fy reduction with alpha
    tire_r_by3 = params[69]  # Shift term for alpha in slope Fy reduction
    tire_r_cy1 = params[70]  # Shape factor for combined Fy reduction
    tire_r_ey1 = params[71]  # Curvature factor of combined Fy
    tire_r_hy1 = params[72]  # Shift factor for combined Fy reduction
    tire_r_vy1 = params[73]  # Kappa induced side force Svyk/Muy*Fz at Fznom
    tire_r_vy3 = params[74]  # Variation of Svyk/Muy*Fz with camber
    tire_r_vy4 = params[75]  # Variation of Svyk/Muy*Fz with alpha
    tire_r_vy5 = params[76]  # Variation of Svyk/Muy*Fz with kappa
    tire_r_vy6 = params[77]  # Variation of Svyk/Muy*Fz with atan(kappa)


    #consider steering constraints
    u = []
    u.append(steering_constraints(x[2], uInit[0], p.steering)) # different name u_init/u due to side effects of u
    #consider acceleration constraints
    u.append(acceleration_constraints(x[3], uInit[1], p.longitudinal)) # different name u_init/u due to side effects of u

    #compute slip angle at cg
    #switch to kinematic model for small velocities
    if abs(x[3]) < 0.1:
        beta = 0.
    else:
        beta = math.atan(x[10]/x[3])
    vel = math.sqrt(x[3]**2 + x[10]**2)



    #vertical tire forces
    F_z_LF = (x[16] + p.R_w*(math.cos(x[13]) - 1) - 0.5*p.T_f*math.sin(x[13]))*p.K_zt
    F_z_RF = (x[16] + p.R_w*(math.cos(x[13]) - 1) + 0.5*p.T_f*math.sin(x[13]))*p.K_zt
    F_z_LR = (x[21] + p.R_w*(math.cos(x[18]) - 1) - 0.5*p.T_r*math.sin(x[18]))*p.K_zt
    F_z_RR = (x[21] + p.R_w*(math.cos(x[18]) - 1) + 0.5*p.T_r*math.sin(x[18]))*p.K_zt

    #obtain individual tire speeds
    u_w_lf = (x[3] + 0.5*p.T_f*x[5])*math.cos(x[2]) + (x[10] + p.a*x[5])*math.sin(x[2])
    u_w_rf = (x[3] - 0.5*p.T_f*x[5])*math.cos(x[2]) + (x[10] + p.a*x[5])*math.sin(x[2])
    u_w_lr = x[3] + 0.5*p.T_r*x[5]
    u_w_rr = x[3] - 0.5*p.T_r*x[5]

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
        s_lf = 1 - p.R_w*x[23]/u_w_lf
        s_rf = 1 - p.R_w*x[24]/u_w_rf
        s_lr = 1 - p.R_w*x[25]/u_w_lr
        s_rr = 1 - p.R_w*x[26]/u_w_rr

        #lateral slip angles
    #switch to kinematic model for small velocities
    if abs(x[3]) < 0.1:
        alpha_LF = 0.
        alpha_RF = 0.
        alpha_LR = 0.
        alpha_RR = 0.
    else:
        alpha_LF = math.atan((x[10] + p.a*x[5] - x[14]*(p.R_w - x[16]))/(x[3] + 0.5*p.T_f*x[5])) - x[2]
        alpha_RF = math.atan((x[10] + p.a*x[5] - x[14]*(p.R_w - x[16]))/(x[3] - 0.5*p.T_f*x[5])) - x[2]
        alpha_LR = math.atan((x[10] - p.b*x[5] - x[19]*(p.R_w - x[21]))/(x[3] + 0.5*p.T_r*x[5]))
        alpha_RR = math.atan((x[10] - p.b*x[5] - x[19]*(p.R_w - x[21]))/(x[3] - 0.5*p.T_r*x[5]))

        #auxiliary suspension movement
    z_SLF = (p.h_s - p.R_w + x[16] - x[11])/math.cos(x[6]) - p.h_s + p.R_w + p.a*x[8] + 0.5*(x[6] - x[13])*p.T_f
    z_SRF = (p.h_s - p.R_w + x[16] - x[11])/math.cos(x[6]) - p.h_s + p.R_w + p.a*x[8] - 0.5*(x[6] - x[13])*p.T_f
    z_SLR = (p.h_s - p.R_w + x[21] - x[11])/math.cos(x[6]) - p.h_s + p.R_w - p.b*x[8] + 0.5*(x[6] - x[18])*p.T_r
    z_SRR = (p.h_s - p.R_w + x[21] - x[11])/math.cos(x[6]) - p.h_s + p.R_w - p.b*x[8] - 0.5*(x[6] - x[18])*p.T_r

    dz_SLF = x[17] - x[12] + p.a*x[9] + 0.5*(x[7] - x[14])*p.T_f
    dz_SRF = x[17] - x[12] + p.a*x[9] - 0.5*(x[7] - x[14])*p.T_f
    dz_SLR = x[22] - x[12] - p.b*x[9] + 0.5*(x[7] - x[19])*p.T_r
    dz_SRR = x[22] - x[12] - p.b*x[9] - 0.5*(x[7] - x[19])*p.T_r

    #camber angles
    gamma_LF = x[6] + p.D_f*z_SLF + p.E_f*(z_SLF)**2
    gamma_RF = x[6] - p.D_f*z_SRF - p.E_f*(z_SRF)**2
    gamma_LR = x[6] + p.D_r*z_SLR + p.E_r*(z_SLR)**2
    gamma_RR = x[6] - p.D_r*z_SRR - p.E_r*(z_SRR)**2

    #compute longitudinal tire forces using the magic formula for pure slip
    F0_x_LF = tireModel.formula_longitudinal(s_lf, gamma_LF, F_z_LF, p.tire)
    F0_x_RF = tireModel.formula_longitudinal(s_rf, gamma_RF, F_z_RF, p.tire)
    F0_x_LR = tireModel.formula_longitudinal(s_lr, gamma_LR, F_z_LR, p.tire)
    F0_x_RR = tireModel.formula_longitudinal(s_rr, gamma_RR, F_z_RR, p.tire)

    #compute lateral tire forces using the magic formula for pure slip
    res = tireModel.formula_lateral(alpha_LF, gamma_LF, F_z_LF, p.tire)
    F0_y_LF = res[0]
    mu_y_LF = res[1]
    res = tireModel.formula_lateral(alpha_RF, gamma_RF, F_z_RF, p.tire)
    F0_y_RF = res[0]
    mu_y_RF = res[1]
    res = tireModel.formula_lateral(alpha_LR, gamma_LR, F_z_LR, p.tire)
    F0_y_LR = res[0]
    mu_y_LR = res[1]
    res = tireModel.formula_lateral(alpha_RR, gamma_RR, F_z_RR, p.tire)
    F0_y_RR = res[0]
    mu_y_RR = res[1]

    #compute longitudinal tire forces using the magic formula for combined slip
    F_x_LF = tireModel.formula_longitudinal_comb(s_lf, alpha_LF, F0_x_LF, p.tire)
    F_x_RF = tireModel.formula_longitudinal_comb(s_rf, alpha_RF, F0_x_RF, p.tire)
    F_x_LR = tireModel.formula_longitudinal_comb(s_lr, alpha_LR, F0_x_LR, p.tire)
    F_x_RR = tireModel.formula_longitudinal_comb(s_rr, alpha_RR, F0_x_RR, p.tire)

    #compute lateral tire forces using the magic formula for combined slip
    F_y_LF = tireModel.formula_lateral_comb(s_lf, alpha_LF, gamma_LF, mu_y_LF, F_z_LF, F0_y_LF, p.tire)
    F_y_RF = tireModel.formula_lateral_comb(s_rf, alpha_RF, gamma_RF, mu_y_RF, F_z_RF, F0_y_RF, p.tire)
    F_y_LR = tireModel.formula_lateral_comb(s_lr, alpha_LR, gamma_LR, mu_y_LR, F_z_LR, F0_y_LR, p.tire)
    F_y_RR = tireModel.formula_lateral_comb(s_rr, alpha_RR, gamma_RR, mu_y_RR, F_z_RR, F0_y_RR, p.tire)

    #auxiliary movements for compliant joint equations
    delta_z_f = p.h_s - p.R_w + x[16] - x[11]
    delta_z_r = p.h_s - p.R_w + x[21] - x[11]

    delta_phi_f = x[6] - x[13]
    delta_phi_r = x[6] - x[18]

    dot_delta_phi_f = x[7] - x[14]
    dot_delta_phi_r = x[7] - x[19]

    dot_delta_z_f = x[17] - x[12]
    dot_delta_z_r = x[22] - x[12]

    dot_delta_y_f = x[10] + p.a*x[5] - x[15]
    dot_delta_y_r = x[10] - p.b*x[5] - x[20]

    delta_f = delta_z_f*math.sin(x[6]) - x[27]*math.cos(x[6]) - (p.h_raf - p.R_w)*math.sin(delta_phi_f)
    delta_r = delta_z_r*math.sin(x[6]) - x[28]*math.cos(x[6]) - (p.h_rar - p.R_w)*math.sin(delta_phi_r)

    dot_delta_f = (delta_z_f*math.cos(x[6]) + x[27]*math.sin(x[6]))*x[7] + dot_delta_z_f*math.sin(x[6]) - dot_delta_y_f*math.cos(x[6]) - (p.h_raf - p.R_w)*math.cos(delta_phi_f)*dot_delta_phi_f
    dot_delta_r = (delta_z_r*math.cos(x[6]) + x[28]*math.sin(x[6]))*x[7] + dot_delta_z_r*math.sin(x[6]) - dot_delta_y_r*math.cos(x[6]) - (p.h_rar - p.R_w)*math.cos(delta_phi_r)*dot_delta_phi_r

    #compliant joint forces
    F_RAF = delta_f*p.K_ras + dot_delta_f*p.K_rad
    F_RAR = delta_r*p.K_ras + dot_delta_r*p.K_rad

    #auxiliary suspension forces (bump stop neglected  squat/lift forces neglected)
    F_SLF = p.m_s*g*p.b/(2*(p.a+p.b)) - z_SLF*p.K_sf - dz_SLF*p.K_sdf + (x[6] - x[13])*p.K_tsf/p.T_f

    F_SRF = p.m_s*g*p.b/(2*(p.a+p.b)) - z_SRF*p.K_sf - dz_SRF*p.K_sdf - (x[6] - x[13])*p.K_tsf/p.T_f

    F_SLR = p.m_s*g*p.a/(2*(p.a+p.b)) - z_SLR*p.K_sr - dz_SLR*p.K_sdr + (x[6] - x[18])*p.K_tsr/p.T_r

    F_SRR = p.m_s*g*p.a/(2*(p.a+p.b)) - z_SRR*p.K_sr - dz_SRR*p.K_sdr - (x[6] - x[18])*p.K_tsr/p.T_r


    #auxiliary variables sprung mass
    sumX = F_x_LR + F_x_RR + (F_x_LF + F_x_RF)*math.cos(x[2]) - (F_y_LF + F_y_RF)*math.sin(x[2])

    sumN = (F_y_LF + F_y_RF)*p.a*math.cos(x[2]) + (F_x_LF + F_x_RF)*p.a*math.sin(x[2]) \
           + (F_y_RF - F_y_LF)*0.5*p.T_f*math.sin(x[2]) + (F_x_LF - F_x_RF)*0.5*p.T_f*math.cos(x[2]) \
           + (F_x_LR - F_x_RR)*0.5*p.T_r - (F_y_LR + F_y_RR)*p.b

    sumY_s = (F_RAF + F_RAR)*math.cos(x[6]) + (F_SLF + F_SLR + F_SRF + F_SRR)*math.sin(x[6])

    sumL = 0.5*F_SLF*p.T_f + 0.5*F_SLR*p.T_r - 0.5*F_SRF*p.T_f - 0.5*F_SRR*p.T_r \
           - F_RAF/math.cos(x[6])*(p.h_s - x[11] - p.R_w + x[16] - (p.h_raf - p.R_w)*math.cos(x[13])) \
           - F_RAR/math.cos(x[6])*(p.h_s - x[11] - p.R_w + x[21] - (p.h_rar - p.R_w)*math.cos(x[18]))

    sumZ_s = (F_SLF + F_SLR + F_SRF + F_SRR)*math.cos(x[6]) - (F_RAF + F_RAR)*math.sin(x[6])

    sumM_s = p.a*(F_SLF + F_SRF) - p.b*(F_SLR + F_SRR) + ((F_x_LF + F_x_RF)*math.cos(x[2]) \
                                                          - (F_y_LF + F_y_RF)*math.sin(x[2]) + F_x_LR + F_x_RR)*(p.h_s - x[11])

    #auxiliary variables unsprung mass
    sumL_uf = 0.5*F_SRF*p.T_f - 0.5*F_SLF*p.T_f - F_RAF*(p.h_raf - p.R_w) \
              + F_z_LF*(p.R_w*math.sin(x[13]) + 0.5*p.T_f*math.cos(x[13]) - p.K_lt*F_y_LF) \
              - F_z_RF*(-p.R_w*math.sin(x[13]) + 0.5*p.T_f*math.cos(x[13]) + p.K_lt*F_y_RF) \
              - ((F_y_LF + F_y_RF)*math.cos(x[2]) + (F_x_LF + F_x_RF)*math.sin(x[2]))*(p.R_w - x[16])

    sumL_ur = 0.5*F_SRR*p.T_r - 0.5*F_SLR*p.T_r - F_RAR*(p.h_rar - p.R_w) \
              + F_z_LR*(p.R_w*math.sin(x[18]) + 0.5*p.T_r*math.cos(x[18]) - p.K_lt*F_y_LR) \
              - F_z_RR*(-p.R_w*math.sin(x[18]) + 0.5*p.T_r*math.cos(x[18]) + p.K_lt*F_y_RR) \
              - (F_y_LR + F_y_RR)*(p.R_w - x[21])

    sumZ_uf = F_z_LF + F_z_RF + F_RAF*math.sin(x[6]) - (F_SLF + F_SRF)*math.cos(x[6])

    sumZ_ur = F_z_LR + F_z_RR + F_RAR*math.sin(x[6]) - (F_SLR + F_SRR)*math.cos(x[6])

    sumY_uf = (F_y_LF + F_y_RF)*math.cos(x[2]) + (F_x_LF + F_x_RF)*math.sin(x[2]) \
              - F_RAF*math.cos(x[6]) - (F_SLF + F_SRF)*math.sin(x[6])

    sumY_ur = (F_y_LR + F_y_RR) \
              - F_RAR*math.cos(x[6]) - (F_SLR + F_SRR)*math.sin(x[6])


    #dynamics common with single-track model
    f = [] # init 'right hand side'
    #switch to kinematic model for small velocities
    if abs(x[3]) < 0.1:
        #wheelbase
        # lwb = p.a + p.b

        #system dynamics
        # x_ks = [x[0],  x[1],  x[2],  x[3],  x[4]]
        # f_ks = vehicle_dynamics_ks(x_ks, u, p)
        # f.extend(f_ks)
        # f.append(u[1]*lwb*math.tan(x[2]) + x[3]/(lwb*math.cos(x[2])**2)*u[0])

        # Use kinematic model with reference point at center of mass
        # wheelbase
        lwb = p.a + p.b
        # system dynamics
        x_ks = [x[0], x[1], x[2], x[3], x[4]]
        # kinematic model
        f_ks = vehicle_dynamics_ks_cog(x_ks, u, p)
        f = [f_ks[0], f_ks[1], f_ks[2], f_ks[3], f_ks[4]]
        # derivative of slip angle and yaw rate
        d_beta = (p.b * u[0]) / (lwb * math.cos(x[2]) ** 2 * (1 + (math.tan(x[2]) ** 2 * p.b / lwb) ** 2))
        dd_psi = 1 / lwb * (u[1] * math.cos(x[6]) * math.tan(x[2]) -
                            x[3] * math.sin(x[6]) * d_beta * math.tan(x[2]) +
                            x[3] * math.cos(x[6]) * u[0] / math.cos(x[2]) ** 2)
        f.append(dd_psi)

    else:
        f.append(math.cos(beta + x[4])*vel)
        f.append(math.sin(beta + x[4])*vel)
        f.append(u[0])
        f.append(1/p.m*sumX + x[5]*x[10])
        f.append(x[5])
        f.append(1/(p.I_z - (p.I_xz_s)**2/p.I_Phi_s)*(sumN + p.I_xz_s/p.I_Phi_s*sumL))


    # remaining sprung mass dynamics
    f.append(x[7])
    f.append(1/(p.I_Phi_s - (p.I_xz_s)**2/p.I_z)*(p.I_xz_s/p.I_z*sumN + sumL))
    f.append(x[9])
    f.append(1/p.I_y_s*sumM_s)
    f.append(1/p.m_s*sumY_s - x[5]*x[3])
    f.append(x[12])
    f.append(g - 1/p.m_s*sumZ_s)

    #unsprung mass dynamics (front)
    f.append(x[14])
    f.append(1/p.I_uf*sumL_uf)
    f.append(1/p.m_uf*sumY_uf - x[5]*x[3])
    f.append(x[17])
    f.append(g - 1/p.m_uf*sumZ_uf)

    #unsprung mass dynamics (rear)
    f.append(x[19])
    f.append(1/p.I_ur*sumL_ur)
    f.append(1/p.m_ur*sumY_ur - x[5]*x[3])
    f.append(x[22])
    f.append(g - 1/p.m_ur*sumZ_ur)

    #convert acceleration input to brake and engine torque
    if u[1]>0:
        T_B = 0.0
        T_E = p.m*p.R_w*u[1]
    else:
        T_B = p.m*p.R_w*u[1]
        T_E = 0.



    #wheel dynamics (p.T  new parameter for torque splitting)
    f.append(1/p.I_y_w*(-p.R_w*F_x_LF + 0.5*p.T_sb*T_B + 0.5*p.T_se*T_E))
    f.append(1/p.I_y_w*(-p.R_w*F_x_RF + 0.5*p.T_sb*T_B + 0.5*p.T_se*T_E))
    f.append(1/p.I_y_w*(-p.R_w*F_x_LR + 0.5*(1-p.T_sb)*T_B + 0.5*(1-p.T_se)*T_E))
    f.append(1/p.I_y_w*(-p.R_w*F_x_RR + 0.5*(1-p.T_sb)*T_B + 0.5*(1-p.T_se)*T_E))

    #negative wheel spin forbidden
    for iState in range(23, 27):
        if x[iState] < 0.0:
            x[iState] = 0.0
            f[iState] = 0.0

    #compliant joint equations
    f.append(dot_delta_y_f)
    f.append(dot_delta_y_r)

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