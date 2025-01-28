import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
import math

# Friction model parameters
viscosity = 0.0036  # [Ns/m2]
r_barrel = 0.00318  # [m]
l_stop = 0.007  # [m]
thick = 1.5e-9  # [m]
needle_length = 0.02  # [m]
needle_radius = 0.00025  # [m]

def friction_mechanistic_model(speed):
    """
    Mechanistic model for friction force.

    Parameters:
        speed (float): Piston speed [m/s].

    Returns:
        float: Friction force [N].
    """
    friction_force = ((2 * math.pi * viscosity * r_barrel * l_stop) / thick) * speed
    return friction_force

def hydrodynamic_force(speed):
    """
    Mechanistic model for hydrodynamic force.

    Parameters:
        speed (float): Piston speed [m/s].

    Returns:
        float: Hydrodynamic force [N].
    """
    radius_barrel = r_barrel
    prod_viscosity = viscosity
    hydro_force = (((8 * math.pi * prod_viscosity * needle_length * radius_barrel**4) / (needle_radius**2)) * speed * 1000)
    return hydro_force

def piston_simulation_with_injection_time(m, k, x0, F_hydro_func, F_friction_func, x_init, v_init, x_final, t_span):
    """
    Simulates the piston motion and calculates the injection time.

    Parameters:
        m (float): Mass of the piston [kg].
        k (float): Spring constant [N/m].
        x0 (float): Rest position of the spring [m].
        F_hydro_func (function): Function to calculate hydrodynamic force [N].
        F_friction_func (function): Function to calculate friction force [N].
        x_init (float): Initial position of the piston [m].
        v_init (float): Initial velocity of the piston [m/s].
        x_final (float): Final position of the piston [m] (when injection is complete).
        t_span (tuple): Time interval for the simulation [s].

    Returns:
        tuple: Simulation results and injection time.
    """
    def equations(t, y):
        x, v = y
        F_spring = -k * (x - x0)
        F_hydro = F_hydro_func(v)
        F_friction = F_friction_func(v)
        F_net = F_spring - F_hydro - F_friction
        a = F_net / m  # Acceleration
        return [v, a]

    # Solve the ODE
    solution = solve_ivp(
        equations, t_span, [x_init, v_init], method='RK45', max_step=0.01, events=lambda t, y: y[0] - x_final
    )

    # Extract the injection time from the event
    if solution.t_events[0].size > 0:
        injection_time = solution.t_events[0][0]  # Time at which x_final is reached
    else:
        injection_time = None  # If x_final is never reached in t_span

    return solution, injection_time

# Parameters
m = 0.05       # Mass of piston [kg]
k = 500.0      # Spring constant [N/m]
x0 = 0.0       # Rest position of the spring [m]
x_init = 0.034 # Initial position of the piston [m]
x_final = 0.0  # Final position of the piston [m]
v_init = 0.0   # Initial velocity [m/s]
t_span = (0, 15.0)  # Extended time span [s]

# Simulate the piston motion and calculate injection time
solution, injection_time = piston_simulation_with_injection_time(
    m, k, x0, hydrodynamic_force, friction_mechanistic_model, x_init, v_init, x_final, t_span
)

# Calculate average velocity
if injection_time is not None:
    total_distance = x_init - x_final
    avg_velocity = total_distance / injection_time
    print(f"Injection time: {injection_time:.2f} s")
    print(f"Average velocity: {avg_velocity:.5f} m/s")
else:
    print("The piston did not reach the final position within the specified time.")

# Plot results
plt.figure(figsize=(10, 6))
plt.plot(solution.t, solution.y[0], label="Position (m)")
plt.axhline(x_init, color="green", linestyle="--", label="Initial Position")
plt.axhline(x_final, color="blue", linestyle="--", label="Final Position")
plt.xlabel("Time (s)")
plt.ylabel("Position (m)")
plt.title("Position of the Piston Over Time")
plt.grid()
plt.legend()
plt.show()

# Plot elastic force over time
F_spring = -k * (solution.y[0] - x0)
plt.figure(figsize=(10, 6))
plt.plot(solution.t, F_spring, label="Elastic Force (N)", color="orange")
plt.xlabel("Time (s)")
plt.ylabel("Force (N)")
plt.title("Elastic Force Over Time")
plt.grid()
plt.legend()
plt.show()

# Plot friction force
F_friction = np.array([friction_mechanistic_model(v) for v in solution.y[1]])
plt.figure(figsize=(10, 6))
plt.plot(solution.t, F_friction, label="Friction Force (N)", color="red")
plt.xlabel("Time (s)")
plt.ylabel("Force (N)")
plt.title("Friction Force Over Time")
plt.grid()
plt.legend()
plt.show()

# Plot hydrodynamic force
F_hydro = np.array([hydrodynamic_force(v) for v in solution.y[1]])
plt.figure(figsize=(10, 6))
plt.plot(solution.t, F_hydro, label="Hydrodynamic Force (N)", color="blue")
plt.xlabel("Time (s)")
plt.ylabel("Force (N)")
plt.title("Hydrodynamic Force Over Time")
plt.grid()
plt.legend()
plt.show()

