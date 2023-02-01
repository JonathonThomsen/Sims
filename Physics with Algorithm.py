import math
import random
import matplotlib.pyplot as plt
import numpy as np
import pandas


# Hope this works

def thrustcurve(engine_file):
    """
    This function takes the engine_file.csv argument (from thrustcurve.org), reads the data, and creates lists of
    thrust values, the times at which those values occur, and calculated specific impulses.
    :param engine_file:
    :return:
    """
    reader = pandas.read_csv(engine_file)
    thrust_list = reader.values.tolist()
    thrust_list = thrust_list[3:len(thrust_list)]
    thrust_times = [float(thrust_list[i][0]) for i in range(len(thrust_list))]
    thrusts_curve = [float(thrust_list[i][1]) * 0.2248089431 for i in range(len(thrust_list))]
    ISP = (np.trapz(thrusts_curve, thrust_times) / (wet_mass - dry_mass)) / np.interp(start_elevation, altitudes,
                                                                                      gravities)
    fuel_flow = [(thrust / (ISP * gravity)) for thrust in thrusts_curve]
    return thrust_times, thrusts_curve, fuel_flow


def control_logic(alldata):
    """
    This function takes all the data from the previous time step to determine if the airbrakes should be deployed.
    The function returns a boolean for controlling the airbrakes and one data point that is used to determine if the
    airbrakes should be deployed.
    :param alldata:
    :return:
    """
    apply = False
    if apply:
        expected_apogee = -((alldata[2] ** 2) / (2 * alldata[1])) + (alldata[3] - start_elevation)

        if ((expected_apogee - alldata[3]) / expected_apogee) >0:

            return True, expected_apogee

        else:
            return False, expected_apogee
    else:
        return False, 0


def iterative_tunnel(previous):
    """
    The big daddy of this program. The function takes all the startup parameters and iteratively finds data for each
    moment, dt, in the rocket's flight.
    :param previous:
    :return:
    """
    # UNPACKING __________________________________________________________________
    flight_time, z_acceleration, z_velocity, z_position, y_acceleration, y_velocity, y_position, x_acceleration, \
    x_velocity, x_position, spin_velocity, spin_position, thrust, mass, \
    mass_flow_rate, dt, drag_momentum, time_to_apogee, deployment_angle, deploy_brakes = previous

    # ENVIRONMENT DATA ___________________________________________________________
    local_gravity = np.interp(z_position, altitudes, gravities)
    local_density = np.interp(z_position, altitudes, densities)
    wind_x, wind_y, spin_accel = environment(wind, z_position)

    if deploy_brakes:
        if deployment_angle < 90:
            checker = deployment_angle + (90 * dt / airbrake_deployment_time)
            if checker > 90:
                deployment_angle = 90
            else:
                deployment_angle = checker

    elif not deploy_brakes:
        if deployment_angle > 0:
            checker = deployment_angle - (90 * dt / airbrake_deployment_time)
            if checker < 0:
                deployment_angle = 0
            else:
                deployment_angle = checker

    if z_position < start_elevation:
        return
    else:
        current = []
        flight_time = flight_time + dt
        if mass > dry_mass:
            mass_flow_rate = np.interp(flight_time, times, mass_flow_rates)
            mass = mass - (mass_flow_rate * dt)
            thrust = np.interp(flight_time, times, thrusts)

        else:
            mass = dry_mass
            mass_flow_rate = 0
            thrust = 0
        z_acceleration = ((thrust * math.cos(launch_rail_phi) / mass) - local_gravity)
        total_vel = math.sqrt(x_velocity ** 2 + y_velocity ** 2 + z_velocity ** 2)
        airbrake_area = (29 / 144) * math.sin(
            deployment_angle * math.pi / 180)
        airbrake_cd = 1.17 * math.sin(deployment_angle * math.pi / 180)
        if z_velocity > 100:
            flight_cd = drag_coefficient(deployment_angle, z_position, z_velocity)
        else:
            flight_cd = cd + airbrake_cd
        if z_velocity >= 0:
            drag_momentum = -(.5 * flight_cd
                              * local_density
                              * (total_vel ** 2)
                              * (frontal_area + airbrake_area)
                              * dt)
        else:
            drag_momentum = (.5 * flight_cd
                             * local_density
                             * (total_vel ** 2)
                             * (frontal_area + airbrake_area)
                             * dt)

            # print(drag/dt)
        z_velocity = z_velocity + (z_acceleration * dt) + (drag_momentum / mass)
        z_position = z_position + (z_velocity * dt) + (.5 * z_acceleration * (dt ** 2))
        if thrust == 0:
            y_acceleration = 0
            x_acceleration = 0
        else:
            y_acceleration = (z_acceleration * math.sin(launch_rail_phi) * math.sin(launch_rail_theta))
            x_acceleration = z_acceleration * math.sin(launch_rail_phi) * math.cos(launch_rail_theta)
        if wind_y < 0 or wind_y > 0:
            y_velocity = y_velocity + (y_acceleration * dt) + (1 - (y_velocity / wind_y))
        if wind_x < 0 or wind_x > 0:
            x_velocity = x_velocity + (x_acceleration * dt) + (1 - (x_velocity / wind_x))
        else:
            y_velocity = y_velocity + (y_acceleration * dt)
            x_velocity = x_velocity + (x_acceleration * dt)
        y_position = y_position + (y_velocity * dt) + (.5 * y_acceleration * (dt ** 2))
        x_position = x_position + (x_velocity * dt) + (.5 * x_acceleration * (dt ** 2))
        spin_velocity = spin_velocity + (spin_accel * dt)
        spin_position = (spin_position + (spin_velocity * dt) + (.5 * spin_accel * (dt ** 2)))
        current.append(flight_time)
        current.append(z_acceleration)
        current.append(z_velocity)
        current.append(z_position)
        current.append(y_acceleration)
        current.append(y_velocity)
        current.append(y_position)
        current.append(x_acceleration)
        current.append(x_velocity)
        current.append(x_position)
        current.append(spin_velocity)
        current.append(spin_position)
        current.append(thrust)
        current.append(mass)
        current.append(mass_flow_rate)

        if (feet_per_iteration / z_velocity) > max_dt or (feet_per_iteration / z_velocity) <= 0:
            dt = max_dt
        else:
            dt = feet_per_iteration / z_velocity
        current.append(dt)
        current.append(drag_momentum / dt)

        # print(temp)
        deploy_brakes, expected_apogee = control_logic(current)

        current.append(expected_apogee)
        current.append(deployment_angle)
        current.append(deploy_brakes)

        flight.append(current)
        print(current)
        return current


def recursive_tunnel(previous):
    """
    The big daddy of this program. The function takes all the startup parameters and recursively finds data for each
    moment, dt, in the rocket's flight.
    :param previous:
    :return:
    """
    # UNPACKING __________________________________________________________________
    flight_time, z_acceleration, z_velocity, z_position, y_acceleration, y_velocity, y_position, x_acceleration, \
    x_velocity, x_position, spin_velocity, spin_position, thrust, mass, \
    mass_flow_rate, dt, drag_momentum, time_to_apogee, deployment_angle, deploy_brakes = previous

    # ENVIRONMENT DATA ___________________________________________________________
    local_gravity = np.interp(z_position, altitudes, gravities)
    local_density = np.interp(z_position, altitudes, densities)
    wind_x, wind_y, spin_accel = environment(wind, z_position)

    if deploy_brakes:
        if deployment_angle < 90:
            checker = deployment_angle + (90 * dt / airbrake_deployment_time)
            if checker > 90:
                deployment_angle = 90
            else:
                deployment_angle = checker

    elif not deploy_brakes:
        if deployment_angle > 0:
            checker = deployment_angle - (90 * dt / airbrake_deployment_time)
            if checker < 0:
                deployment_angle = 0
            else:
                deployment_angle = checker

    if z_position < start_elevation:
        return
    else:
        current = []
        flight_time = flight_time + dt
        if mass > dry_mass:
            mass_flow_rate = np.interp(flight_time, times, mass_flow_rates)
            mass = mass - (mass_flow_rate * dt)
            thrust = np.interp(flight_time, times, thrusts)

        else:
            mass = dry_mass
            mass_flow_rate = 0
            thrust = 0
        z_acceleration = ((thrust * math.cos(launch_rail_phi) / mass) - local_gravity)
        total_vel = math.sqrt(x_velocity ** 2 + y_velocity ** 2 + z_velocity ** 2)
        airbrake_area = (29 / 144) * math.sin(
            deployment_angle * math.pi / 180)
        airbrake_cd = 1.17 * math.sin(deployment_angle * math.pi / 180)
        if z_velocity > 100:
            flight_cd = drag_coefficient(deployment_angle, z_position, z_velocity)
        else:
            flight_cd = cd + airbrake_cd
        if z_velocity >= 0:
            drag_momentum = -(.5 * flight_cd
                              * local_density
                              * (total_vel ** 2)
                              * (frontal_area + airbrake_area)
                              * dt)
        else:
            drag_momentum = (.5 * flight_cd
                             * local_density
                             * (total_vel ** 2)
                             * (frontal_area + airbrake_area)
                             * dt)

            # print(drag/dt)
        z_velocity = z_velocity + (z_acceleration * dt) + (drag_momentum / mass)
        z_position = z_position + (z_velocity * dt) + (.5 * z_acceleration * (dt ** 2))
        if thrust == 0:
            y_acceleration = 0
            x_acceleration = 0
        else:
            y_acceleration = (z_acceleration * math.sin(launch_rail_phi) * math.sin(launch_rail_theta))
            x_acceleration = z_acceleration * math.sin(launch_rail_phi) * math.cos(launch_rail_theta)
        if wind_y < 0 or wind_y > 0:
            y_velocity = y_velocity + (y_acceleration * dt) + (1 - (y_velocity / wind_y))
        if wind_x < 0 or wind_x > 0:
            x_velocity = x_velocity + (x_acceleration * dt) + (1 - (x_velocity / wind_x))
        else:
            y_velocity = y_velocity + (y_acceleration * dt)
            x_velocity = x_velocity + (x_acceleration * dt)
        y_position = y_position + (y_velocity * dt) + (.5 * y_acceleration * (dt ** 2))
        x_position = x_position + (x_velocity * dt) + (.5 * x_acceleration * (dt ** 2))
        spin_velocity = spin_velocity + (spin_accel * dt)
        spin_position = (spin_position + (spin_velocity * dt) + (.5 * spin_accel * (dt ** 2)))
        current.append(flight_time)
        current.append(z_acceleration)
        current.append(z_velocity)
        current.append(z_position)
        current.append(y_acceleration)
        current.append(y_velocity)
        current.append(y_position)
        current.append(x_acceleration)
        current.append(x_velocity)
        current.append(x_position)
        current.append(spin_velocity)
        current.append(spin_position)
        current.append(thrust)
        current.append(mass)
        current.append(mass_flow_rate)

        if (feet_per_iteration / z_velocity) > max_dt or (feet_per_iteration / z_velocity) <= 0:
            dt = max_dt
        else:
            dt = feet_per_iteration / z_velocity
        current.append(dt)
        current.append(drag_momentum / dt)

        # print(temp)
        deploy_brakes, expected_apogee = control_logic(current)

        current.append(expected_apogee)
        current.append(deployment_angle)
        current.append(deploy_brakes)

        flight.append(current)
        print(current)
        recursive_tunnel(current)


def drag_coefficient(deployment_angle, altitude, z_velocity):
    """
    Calculates drag coefficient using the linear relationship between the nondimensional "Thomsen Number"
    and the drag force obtained from CFD simulations.
    :param deployment_angle:
    :param altitude:
    :param z_velocity:
    :return:
    """
    airbrake_area = (29 / 144) * math.sin(
        deployment_angle * math.pi / 180)
    total_area = frontal_area + airbrake_area
    local_density = np.interp(altitude, altitudes, densities)
    local_viscosity = np.interp(altitude, altitudes, viscosities)
    TN = total_area * (local_density ** 2) * (z_velocity ** 2) / (local_viscosity ** 2)

    if 0 <= deployment_angle < 30:
        low = deployment_angle-0
        high = 30-deployment_angle
        estimated_drag = (high *(((10 ** -11) * TN) + 1.7689)/30)+ (low*(((10 ** -11) * TN) + 2.0931)/30)
    elif 30 <= deployment_angle < 60:
        low = deployment_angle-30
        high = 60-deployment_angle
        estimated_drag = (high*(((10 ** -11) * TN) + 2.0931)/30)+(low*(((10 ** -11) * TN) + 2.6925)/30)
    else:
        low = deployment_angle-60
        high = 90-deployment_angle
        estimated_drag = (low*(((10 ** -11) * TN) + 2.6925)/30) + (high*(((10 ** -11) * TN) + 2.0551)/30)

    estimated_cd = (2 * estimated_drag) / (local_density * (z_velocity ** 2) * total_area)
    return estimated_cd


def environment(case, vertical):
    """
    This function defines the wind conditions at the launch site. Three cases are available: dynamic, random, and other.
    Dynamic: The wind velocity can be made a function of the altitude.
    Random: Random wind speeds in the specified range.
    Other: No wind.
    :param case: User-inputted to specify if wind should be considered.
    :param vertical: From the program; returns the instantaneous altitude for finding conditions from the "dynamic" case
    :return:
    """
    rotational_accel = random.randrange(-36, 36)
    if case == "dynamic":
        x_wind_velocity = 100 * (vertical / vertical)  # Some function of altitude agl ("vertical" variable)
        y_wind_velocity = 100  # Some function of altitude agl "vertical" variable)

    elif case == "random":
        x_wind_velocity = random.randint(-20, 20)  # Some function of altitude agl
        y_wind_velocity = random.randint(-20, 20)  # Some function of altitude agl

    else:
        x_wind_velocity = 0
        y_wind_velocity = 0

    return x_wind_velocity, y_wind_velocity, rotational_accel


def output(total_flight):
    max_alt = max([instant[3] - start_elevation for instant in total_flight])
    print("Maximum Altitude Reached: ", max_alt)


def analyze():
    fig = plt.figure()
    plt.rc('font', size=5)
    trajectory = fig.add_subplot(2, 3, 1, projection='3d')
    acc = fig.add_subplot(2, 3, 2)
    vel = fig.add_subplot(2, 3, 3)
    spin = fig.add_subplot(2, 3, 4)
    brake = fig.add_subplot(2, 3, 5)
    incoming = fig.add_subplot(2, 3, 6)

    trajectory.plot([instant[9] for instant in flight], [instant[6] for instant in flight],
                    [step[3] - 4500 for step in flight])
    acc.plot([instant[0] for instant in flight], [instant[1] for instant in flight], label='Acceleration (Z)')
    acc.plot([instant[0] for instant in flight], [instant[4] for instant in flight], label='Acceleration (Y)')
    acc.plot([instant[0] for instant in flight], [instant[7] for instant in flight], label='Acceleration (X)')
    vel.plot([instant[0] for instant in flight], [instant[2] for instant in flight], label='Velocity (Z)')
    vel.plot([instant[0] for instant in flight], [instant[5] for instant in flight], label='Velocity (Y)')
    vel.plot([instant[0] for instant in flight], [instant[8] for instant in flight], label='Velocity (X)')
    spin.plot([instant[0] for instant in flight], [instant[10] for instant in flight], label='Spin Velocity (Deg/s)')
    spin.plot([instant[0] for instant in flight], [(180 * math.cos(instant[11] * math.pi / 180)) for instant in flight],
              label='Spin Position (Deg)')
    brake.plot([instant[0] for instant in flight], [instant[18] for instant in flight])
    incoming.plot([instant[0] for instant in flight], [instant[17] for instant in flight])
    acc.title.set_text("Acceleration (ft/s^2)")
    acc.legend(loc='upper right')

    vel.title.set_text("Velocity (ft/s)")
    vel.legend(loc='upper right')

    spin.title.set_text("Spin")
    spin.legend(loc='upper right')

    brake.title.set_text("Airbrake Deployment Angle (Degrees")
    incoming.title.set_text("Projected Apogee (Feet)")

    plt.show()


# Inputs - Rocket and Mission___________________________________________________________________________________________
body_diameter = 6.17
frontal_area = math.pi * ((body_diameter / 2) ** 2) / 144
cd = 0.294  # Should be a function of Reynold's number
start_elevation = 4500
goal_apogee = 10000 + start_elevation
engine = "AeroTech_M2400T.csv"
launch_rail_phi = 1 * math.pi / 180  # 0-90 Degrees, measured from perfectly vertical.
launch_rail_theta = 0 * math.pi / 180  # 0-360 Degrees, measured from positive x-axis
spin_vel = 0
spin_pos = 0
airbrake_deployment_time = 2  # Seconds

# Environment___________________________________________________________________________________________________________

wind = "static"
altitudes = [1000 * i for i in range(0, 26)]
densities = [0.002377, 0.002308, 0.002241, 0.002175, 0.002111,
             0.002048, 0.001987, 0.001927, 0.001869, 0.001812,
             0.001756, 0.001701, 0.001648, 0.001596, 0.001546,
             0.001496, 0.001448, 0.001401, 0.001356, 0.001311,
             0.001267, 0.001225, 0.001184, 0.001144, 0.001105,
             0.001066]

viscosities = [3.7372e-7, 3.7172e-7, 3.6971e-7, 3.6770e-7, 3.6568e-7,
               3.6366e-7, 3.6162e-7, 3.5958e-7, 3.5754e-7, 3.5549e-7,
               3.5343e-7, 3.5136e-7, 3.4928e-7, 3.4720e-7, 3.4512e-7,
               3.4302e-7, 3.4092e-7, 3.3881e-7, 3.3669e-7, 3.3457e-7,
               3.3244e-7, 3.3030e-7, 3.2815e-7, 3.2599e-7, 3.2383e-7,
               3.2166e-7]

gravities = [32.174, 32.171, 32.1679, 32.1648, 32.1617,
             32.1586, 32.1555, 32.1525, 32.1494, 32.1463,
             32.1432, 32.1401, 32.1371, 32.134, 32.1309,
             32.1278, 32.1247, 32.1217, 32.1186, 32.1155,
             32.1124, 32.1094, 32.1063, 32.1032, 32.1001,
             32.0971]

# Starting Conditions___________________________________________________________________________________________________
gravity = np.interp(start_elevation, altitudes, gravities)
wet_mass = 54 / gravity
dry_mass = 38 / gravity
times, thrusts, mass_flow_rates = thrustcurve(engine)
angle = 0
time = 0
# Vertical
z_accel = 0
z_vel = 0
z_pos = start_elevation
z_mom = 0
drag = 0
# X-Direction
x_accel = 0
x_vel = 0
x_pos = 0
# Y-Direction
y_accel = 0
y_vel = 0
y_pos = 0

# Computation Parameters and Startup____________________________________________________________________________________
comp_type = "iterative"
flight = []

# Main Execution Code___________________________________________________________________________________________________
if comp_type == "iterative":
    feet_per_iteration = 1
    max_dt = .01
    temp = [time, z_accel, z_vel, z_pos, y_accel, y_vel, y_pos, x_accel, x_vel, x_pos, spin_vel, spin_pos, thrusts[0],
            wet_mass, mass_flow_rates[0], max_dt, drag, 0, angle, False]
    print(temp)
    flight.append(temp)
    while temp[3] >= start_elevation:
        previous_step = temp
        next_step = iterative_tunnel(temp)
        temp = next_step
        print(temp)
elif comp_type == "recursive":
    feet_per_iteration = 75
    max_dt = .1
    temp = [time, z_accel, z_vel, z_pos, y_accel, y_vel, y_pos, x_accel, x_vel, x_pos, spin_vel, spin_pos, thrusts[0],
            wet_mass, mass_flow_rates[0], max_dt, drag, 0, angle, False]
    print(temp)
    flight.append(temp)
    recursive_tunnel(temp)

output(flight)
analyze()
