import math
import numpy as np
import matplotlib.pyplot as plt
import random
from mpl_toolkits import mplot3d

def u_dot(dt, parameters):
    g, w, fX, theta, r, q, y, z = parameters
    k = (((g/w)*fX) + (g*-math.sin(theta)) + ((r*y)-(q*z)))*dt
    return k
def v_dot(dt, parameters):
    g, w, fY, phi, theta, p, r, z, x = parameters
    k = (((g/w)*fY) + (g*math.sin(phi)*math.cos(theta)) + ((p*z)-(r*x)))*dt
    return k
def w_dot(dt, parameters):
    g, w, fZ, phi, theta, q, p, x, y = parameters
    k = (((g/w)*fZ) + (g*math.cos(phi)*math.cos(theta)) + ((q*x)-(p*y)))*dt
    return k
def p_dot(dt, parameters):
    matrix, mxb, iyyb, izzb, ixzb, q, r, p, mzb, ixxb, iyyb = parameters
    row1 = matrix[0]*(mxb + ((iyyb-izzb)*q*r) + (ixzb*p*q))
    row3 = matrix[2]*(mzb + ((ixxb-iyyb)*p*q)) - (ixzb*q*r)
    k = (row1+row3)*dt
    return k
def q_dot(dt, parameters):
    matrix, myb, izzb, ixxb, ixzb, p, r= parameters
    k = (matrix[1]*(myb+((izzb-ixxb)*p*r)+(ixzb*((r**2)-(p**2)))))*dt
    return k
def r_dot(dt, parameters):
    matrix, mxb, iyyb, izzb, ixzb, q, r, p, mzb, ixxb, iyyb = parameters
    row1 = matrix[0]*(mxb + ((iyyb-izzb)*q*r) + (ixzb*p*q))
    row3 = matrix[2]*(mzb + ((ixxb-iyyb)*p*q)) - (ixzb*q*r)
    k = (row1+row3)*dt
    return k
def xf_dot(dt, parameters):
    theta, psi, phi, x, y, z, vwxf = parameters
    row1 = (math.cos(theta)* math.cos(psi)*x)
    row2 = (((math.sin(phi)*math.sin(theta)*math.cos(psi))-(math.cos(phi)*math.sin(psi)))*y)
    row3 = (((math.cos(phi)* math.sin(theta)* math.cos(psi)) + (math.sin(phi)*math.sin(psi)))*z)
    k = (row1+row2+row3+vwxf)*dt
    return k
def yf_dot(dt, parameters):
    theta, psi, phi, x, y, z, vwyf = parameters
    row1 = (math.cos(theta)* math.sin(psi)*x)
    row2 = (((math.sin(phi)*math.sin(theta)*math.sin(psi))+(math.cos(phi)*math.cos(psi)))*y)
    row3 = (((math.cos(phi)* math.sin(theta)* math.sin(psi)) - (math.sin(phi)*math.cos(psi)))*z)
    k = (row1+row2+row3+vwyf)*dt
    return k
def zf_dot(dt, parameters):
    theta, phi, x, y, z, vwzf = parameters
    row1 = (-math.sin(theta)*x)
    row2 = ((math.sin(phi)*math.cos(theta))*y)
    row3 = ((math.cos(phi)*math.cos(theta))*z)
    k = (row1+row2+row3+vwzf)*dt
    return k
def ba_dot(dt, parameters):
    phi, theta, p, q, r = parameters
    row1 = p
    row2 = (math.sin(phi)*math.sin(theta)/math.cos(theta)) * q
    row3 = (math.cos(phi)*math.sin(theta)/math.cos(theta)) * r
    k = (row1+row2+row3)*dt
    return k
def ea_dot(dt, parameters):
    phi, p, q, r = parameters
    row1 = 0 * p
    row2 = (math.cos(phi)) * q
    row3 = (-math.sin(phi)) * r
    k = (row1+row2+row3)*dt
    return k
def ha_dot(dt, parameters):
    phi, theta, p, q, r = parameters
    row1 = 0 * p
    row2 = (math.sin(phi)/math.cos(theta)) * q
    row3 = (math.cos(phi)/math.cos(theta)) * r
    k = (row1+row2+row3)*dt
    return k

def rk4(integral, val, diff, h):
    diffnumber, UDOT, VDOT, WDOT, PDOT, QDOT, RDOT, XFDOT, YFDOT, ZFDOT, BADOT, EADOT, HADOT = diff
    valnumber, U, V, W, P, Q, R, XF, YF, ZF, BA, EA, HA, = val
    integralnumber, UPOS, VPOS, WPOS, PPOS, QPOS, RPOS, XFPOS, YFPOS, ZFPOS, BAPOS, EAPOS, HAPOS = integral

    # UDOT Calculations_______________________________________________________________________________
    first = [gravity, weight, forceX, EA, R, Q, V, W]
    k1 = u_dot(time, first)
    second = [i+(h*k1/2) for i in first]
    k2 = u_dot(time+(h/2), second)
    third = [i+(h*k2/2) for i in first]
    k3 = u_dot(time+(h/2),third)
    fourth = [i+(h*k3) for i in first]
    k4 = u_dot(time+h, fourth)
    UDOT = ((1/6)*k1*(2*k2)*(2*k3)+k4)
    U = U + (UDOT*h)
    UPOS = UPOS + (U*h) + (.5*UDOT*(h**2))

    # VDOT Calculations_______________________________________________________________________________
    first = [gravity, weight, forceY, BA, EA, P, R, W, U]
    k1 = v_dot(time, first)
    second = [i+(h*k1/2) for i in first]
    k2 = v_dot(time+(h/2), second)
    third = [i+(h*k2/2) for i in first]
    k3 = v_dot(time+(h/2),third)
    fourth = [i+(h*k3) for i in first]
    k4 = v_dot(time+h, fourth)
    VDOT = ((1/6)*k1*(2*k2)*(2*k3)+k4)
    V = V + (VDOT*h)
    VPOS = VPOS + (V*h) + (.5*VDOT*(h**2))

    # WDOT Calculations_______________________________________________________________________________
    first = [gravity, weight, forceZ, BA, EA, Q, P, U, V]
    k1 = w_dot(time, first)
    second = [i+(h*k1/2) for i in first]
    k2 = w_dot(time+(h/2), second)
    third = [i+(h*k2/2) for i in first]
    k3 = w_dot(time+(h/2),third)
    fourth = [i+(h*k3) for i in first]
    k4 = w_dot(time+h, fourth)
    WDOT = ((1/6)*k1*(2*k2)*(2*k3)+k4)
    W = W + (WDOT*h)
    WPOS = WPOS + (W*h) + (.5*WDOT*(h**2))

    # ROTATION RATES__________________________________________________________________________________
    MOI = np.array([[Ixxb, 0, -Ixzb], [0, Iyyb, 0], [-Izxb, 0, Izzb]])
    MOII = np.linalg.inv(MOI)
    # PDOT Calculations_______________________________________________________________________________
    first = [MOII[0], Mxb, Iyyb, Izzb, Ixzb, Q, R, P, Mzb, Ixxb, Iyyb]
    k1 = p_dot(time, first)
    second = [i+(h*k1/2) for i in first]
    k2 = p_dot(time+(h/2), second)
    third = [i+(h*k2/2) for i in first]
    k3 = p_dot(time+(h/2),third)
    fourth = [i+(h*k3) for i in first]
    k4 = p_dot(time+h, fourth)
    PDOT = ((1/6)*k1*(2*k2)*(2*k3)+k4)
    P = P + (PDOT*h)
    PPOS = PPOS + (P*h) + (.5*PDOT*(h**2))

    # QDOT Calculations_______________________________________________________________________________
    first = [MOII[1], Myb, Izzb, Ixxb, Ixzb,P, R]
    k1 = q_dot(time, first)
    second = [i+(h*k1/2) for i in first]
    k2 = q_dot(time+(h/2), second)
    third = [i+(h*k2/2) for i in first]
    k3 = q_dot(time+(h/2),third)
    fourth = [i+(h*k3) for i in first]
    k4 = q_dot(time+h, fourth)
    QDOT = ((1/6)*k1*(2*k2)*(2*k3)+k4)
    Q = Q + (QDOT*h)
    QPOS = QPOS + (Q*h) + (.5*QDOT*(h**2))


    # RDOT Calculations_______________________________________________________________________________
    first = [MOII[2], Mxb, Iyyb, Izzb, Ixzb, Q, R, P, Mzb, Ixxb, Iyyb]
    k1 = r_dot(time, first)
    second = [i+(h*k1/2) for i in first]
    k2 = r_dot(time+(h/2), second)
    third = [i+(h*k2/2) for i in first]
    k3 = r_dot(time+(h/2),third)
    fourth = [i+(h*k3) for i in first]
    k4 = r_dot(time+h, fourth)
    RDOT = ((1/6)*k1*(2*k2)*(2*k3)+k4)
    R = R + (RDOT*h)
    RPOS = RPOS + (R*h) + (.5*RDOT*(h**2))

    # XFDOT Calculations_______________________________________________________________________________
    first = [EA, HA, BA, U, V, W, Vwxf]
    k1 = xf_dot(time, first)
    second = [i+(h*k1/2) for i in first]
    k2 = xf_dot(time+(h/2), second)
    third = [i+(h*k2/2) for i in first]
    k3 = xf_dot(time+(h/2),third)
    fourth = [i+(h*k3) for i in first]
    k4 = xf_dot(time+h, fourth)
    XFDOT = ((1/6)*k1*(2*k2)*(2*k3)+k4)
    XF = XF + (XFDOT*h)
    XFPOS = XFPOS + (XF*h) + (.5*XFDOT*(h**2))


    # YFDOT Calculations_______________________________________________________________________________
    first = [EA, HA, BA, U, V, W, Vwyf]
    k1 = yf_dot(time, first)
    second = [i+(h*k1/2) for i in first]
    k2 = yf_dot(time+(h/2), second)
    third = [i+(h*k2/2) for i in first]
    k3 = yf_dot(time+(h/2),third)
    fourth = [i+(h*k3) for i in first]
    k4 = yf_dot(time+h, fourth)
    YFDOT = ((1/6)*k1*(2*k2)*(2*k3)+k4)
    YF = YF + (YFDOT*h)
    YFPOS = YFPOS + (YF*h) + (.5*YFDOT*(h**2))


    # ZFDOT Calculations_______________________________________________________________________________
    first = [EA, BA, U, V, W, Vwzf]
    k1 = zf_dot(time, first)
    second = [i+(h*k1/2) for i in first]
    k2 = zf_dot(time+(h/2), second)
    third = [i+(h*k2/2) for i in first]
    k3 = zf_dot(time+(h/2),third)
    fourth = [i+(h*k3) for i in first]
    k4 = zf_dot(time+h, fourth)
    ZFDOT = ((1/6)*k1*(2*k2)*(2*k3)+k4)
    ZF = ZF + (ZFDOT*h)
    ZFPOS = ZFPOS + (ZF*h) + (.5*ZFDOT*(h**2))


    # BADOT Calculations_______________________________________________________________________________
    first = [BA, EA, P, Q, R]
    k1 = ba_dot(time, first)
    second = [i+(h*k1/2) for i in first]
    k2 = ba_dot(time+(h/2), second)
    third = [i+(h*k2/2) for i in first]
    k3 = ba_dot(time+(h/2),third)
    fourth = [i+(h*k3) for i in first]
    k4 = ba_dot(time+h, fourth)
    BADOT = ((1/6)*k1*(2*k2)*(2*k3)+k4)
    BA = BA + (BADOT*h)
    BAPOS = BAPOS + (BA*h) + (.5*BADOT*(h**2))


    # EADOT Calculations_______________________________________________________________________________
    first = [BA, P, Q, R]
    k1 = ea_dot(time, first)
    second = [i+(h*k1/2) for i in first]
    k2 = ea_dot(time+(h/2), second)
    third = [i+(h*k2/2) for i in first]
    k3 = ea_dot(time+(h/2),third)
    fourth = [i+(h*k3) for i in first]
    k4 = ea_dot(time+h, fourth)
    EADOT = ((1/6)*k1*(2*k2)*(2*k3)+k4)
    EA = EA + (EADOT*h)
    EAPOS = EAPOS + (EA*h) + (.5*EADOT*(h**2))


    # HADOT Calculations_______________________________________________________________________________
    first = [BA, EA, P, Q, R]
    k1 = ha_dot(time, first)
    second = [i+(h*k1/2) for i in first]
    k2 = ha_dot(time+(h/2), second)
    third = [i+(h*k2/2) for i in first]
    k3 = ha_dot(time+(h/2),third)
    fourth = [i+(h*k3) for i in first]
    k4 = ha_dot(time+h, fourth)
    HADOT = ((1/6)*k1*(2*k2)*(2*k3)+k4)
    HA = HA + (HADOT*h)
    HAPOS = HAPOS + (HA*h) + (.5*HADOT*(h**2))


 

    diffs = [diffnumber+1, UDOT, VDOT, WDOT, PDOT, QDOT, RDOT, XFDOT, YFDOT, ZFDOT, BADOT, EADOT, HADOT]
    vals = [valnumber + 1, U, V, W, P, Q, R, XF, YF, ZF, BA, EA, HA]
    integrals = [integralnumber + 1, UPOS, VPOS, WPOS, PPOS, QPOS, RPOS, XFPOS, YFPOS, ZFPOS, BAPOS, EAPOS, HAPOS]
    return diffs, vals, integrals

    
# IF ISSUES ARISE< DROP THE TIME STEP
# Time
time = 0

# Constants
gravity = -32.2
weight = 54

# Time Step
h = .00001

# Forces
forceX=0
forceY=0
forceZ=0

# Moments of Inertia
Ixxb = 100
Ixyb = 200
Ixzb = 300
Iyxb = 400
Iyyb = 500
Iyzb = 600
Izxb = 700
Izyb = 800
Izzb = 900

# Aerodynamic Moments
Mxb = 1
Myb = 0
Mzb = 0

# Earth-Fixed Components of Wind
Vwxf = 0
Vwyf = 0
Vwzf = 0

# Initial Acceleration Conditions
UDOT = 0
VDOT = 0
WDOT = 0
PDOT = 0
QDOT = 0
RDOT = 0
XFDOT = 0
YFDOT = 0
ZFDOT = 0
BADOT = 0
EADOT = 0
HADOT = 0

# Initial Velocities
U = 0
V = 1
W = 800
P = .01
Q = 0
R = 0
XF = 0
YF = 0
ZF = 0
BA = 0
EA = 0
HA = 0

# Initial Positions
UPOS = 0
VPOS = 0
WPOS = 4500
PPOS = 0
QPOS = 0
RPOS = 0
XFPOS = 0
YFPOS = 0
ZFPOS = 0
BAPOS = 0
EAPOS = 0
HAPOS = 0

totaldifferentials = []
totalvalues = []
totalintegrals = []
differential = [0, UDOT, VDOT, WDOT, PDOT, QDOT, RDOT, XFDOT, YFDOT, ZFDOT, BADOT, EADOT, HADOT]
value = [0, U, V, W, P, Q, R, XF, YF, ZF, BA, EA, HA]
integral = [0, UPOS, VPOS, WPOS, PPOS, QPOS, RPOS, XFPOS, YFPOS, ZFPOS, BAPOS, EAPOS, HAPOS]
print(differential)
print(value)
print(integral)
input()
totalvalues.append(value)
totaldifferentials.append(differential)
totalintegrals.append(integral)
times = [0]
while 0<=integral[3]:
    time = time + h
    differential, value, integral = rk4(integral, value, differential, h)
    totaldifferentials.append(differential)
    totalvalues.append(value)
    totalintegrals.append(integral)
    times.append(time)
    print(integral[3])


labels = ["number", "UDOT", "VDOT", "WDOT", "PDOT", "QDOT", "RDOT", "XFDOT", "YFDOT", "ZFDOT", "BADOT", "EADOT", "HADOT"]
fig, axs = plt.subplots(nrows=3, ncols=4)
iteration = 0
for i in range(0, 3):
    for j in range(0, 4):
        axs[i, j].plot(times[:-1], [totalintegrals[k][iteration] for k in range(0, len(totalintegrals)-1)])
        axs[i, j].set_title(labels[iteration])
        iteration = iteration + 1



plt.show()


newfig = plt.figure()

ax = newfig.add_subplot(111, projection='3d')
u = [totalintegrals[i][1] for i in range(0, len(totalvalues)-1)]
v = [totalintegrals[i][2] for i in range(0, len(totalvalues)-1)]
w = [totalintegrals[i][3] for i in range(0, len(totalvalues)-1)]
ax.plot(u, v, w, 'red')

plt.autoscale(False)
plt.show()



