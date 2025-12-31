import numpy as np
from copy import copy
from ruckig import InputParameter, OutputParameter, Result, Ruckig, Trajectory
from matplotlib import pyplot as plt
from plotly.subplots import make_subplots
import plotly.graph_objects as go
import streamlit as st


class Axis:
    def __init__(self, name, vel, acc, dec, jerk, units):
        self.name = name
        self.vel = vel  # axis nominal speed
        self.acc = acc  # axis nominal acceleration
        self.dec = dec  # axis nominal deceleration
        self.jerk = jerk  # axis nominal jerk
        self.units = units  # units that all values to be according to

    def time_to_perform(self, pos0, pos1, settling_time=0):
        inp = InputParameter(1)
        out = OutputParameter(1)
        inp.current_position = [pos0]
        inp.current_velocity = [0]
        inp.current_acceleration = [0]
        inp.target_position = [pos1]
        inp.target_velocity = [0]
        inp.target_acceleration = [0]
        inp.max_velocity = [self.vel]
        inp.max_acceleration = [self.acc]
        inp.max_jerk = [self.jerk]
        otg = Ruckig(1)
        trajectory = Trajectory(1)
        result = otg.calculate(inp, trajectory)
        if result == Result.ErrorInvalidInput:
            raise Exception('Invalid input!')
        return trajectory.duration + settling_time

    def online_trajectory(self, pos0, pos1, settling_time=0.0):
        position = []
        speed = []
        acc = []
        time = []

        inp = InputParameter(1)
        out = OutputParameter(1)
        inp.current_position = [pos0]
        inp.current_velocity = [0]
        inp.current_acceleration = [0]
        inp.target_position = [pos1]
        inp.target_velocity = [0]
        inp.target_acceleration = [0]
        inp.max_velocity = [self.vel]
        inp.max_acceleration = [self.acc]
        inp.max_jerk = [self.jerk]

        otg = Ruckig(1, 0.001)

        res = Result.Working
        while res == Result.Working:
            res = otg.update(inp, out)
            time.append(out.time)
            position.append(out.new_position)
            speed.append(out.new_velocity)
            acc.append(out.new_acceleration)
            out.pass_to_input(inp)

        return time, position, speed, acc


def generate_plot(start, finish, vel, acc, dec, jerk):
    axis1 = Axis("axis1", vel, acc, dec, jerk, 'mm')
    t1 = axis1.time_to_perform(start, finish)
    time, position, speed, acc = axis1.online_trajectory(start, finish)

    # flatten for 1 DOF
    position = [p[0] for p in position]
    speed = [s[0] for s in speed]
    acc = [a[0] for a in acc]

    fig, ax = plt.subplots(nrows=3, ncols=1, figsize=(24, 20), dpi=100)
    fig.subplots_adjust(hspace=0.4)

    ax[0].plot(time, position, label="Position")
    ax[0].legend()
    ax[0].grid()
    ax[0].set_title("Position")
    ax[0].set_xlabel("Time [sec]")
    ax[0].set_ylabel("[mm]")

    ax[1].plot(time, speed, label="Velocity")
    ax[1].legend()
    ax[1].grid()
    ax[1].set_title("Speed")
    ax[1].set_xlabel("Time [sec]")
    ax[1].set_ylabel("[mm/sec]")

    # fix: use acc here
    ax[2].plot(time, acc, label="Acceleration")
    ax[2].legend()
    ax[2].grid()
    ax[2].set_title("Acceleration")
    ax[2].set_xlabel("Time [sec]")
    ax[2].set_ylabel("[mm/sec2]")

    return t1, fig


def generate_plot2(start, finish, vel, acc, jerk):
    axis1 = Axis("axis1", vel, acc, acc, jerk, 'mm')
    t1 = axis1.time_to_perform(start, finish)
    time, position, speed, acc = axis1.online_trajectory(start, finish)

    # Flatten Y data from [[x]] to [x]
    position = [p[0] for p in position]
    speed = [s[0] for s in speed]
    acc = [a[0] for a in acc]

    # Compute RMS acceleration
    acc_rms = np.sqrt(np.mean(np.square(acc)))

    # Plot 1
    fig1 = go.Figure()
    fig1.add_trace(
        go.Scatter(
            x=time,
            y=position,
            mode="lines",
            hovertemplate="Time: %{x:.3f} s<br>Position: %{y:.3f} uut",
        )
    )
    fig1.update_layout(
        title="Plot of Position",
        xaxis_title="Time [sec]",
        yaxis_title="(User Units)",
        hovermode="x unified",
    )

    # Plot 2
    fig2 = go.Figure()
    fig2.add_trace(
        go.Scatter(
            x=time,
            y=speed,
            mode="lines",
            hovertemplate="Time: %{x:.3f} s<br>Velocity: %{y:.3f} uut/sec",
        )
    )
    fig2.update_layout(
        title="Plot of Velocity",
        xaxis_title="Time [sec]",
        yaxis_title="(User Units)/sec",
        hovermode="x unified",
    )

    # Plot 3
    fig3 = go.Figure()
    fig3.add_trace(
        go.Scatter(
            x=time,
            y=acc,
            mode="lines",
            hovertemplate="Time: %{x:.3f} s<br>Acceleration: %{y:.3f} uut/sec^2",
        )
    )
    fig3.update_layout(
        title="Plot of Acceleration",
        xaxis_title="Time [sec]",
        yaxis_title="(User Units)/sec^2",
        hovermode="x unified",
    )

    return (fig1, fig2, fig3), t1, acc_rms


# Title
st.title("Interactive 3rd Order Motion Profile Generator")

# Sidebar: predefined kinematics + numeric inputs

presets = {
    "IM X axis": {"speed": 1200.0, "acc": 24000.0, "jerk": 600000.0},
    "IM Y axis": {"speed": 1200.0, "acc": 12000.0, "jerk": 185000.0},
    "IM Theta axis": {"speed": 22.0, "acc": 140.0, "jerk": 1500.0},
    "IM Polarizer": {"speed": 53.0, "acc": 3400.0, "jerk": 235000.0}
}

axis_choice = st.sidebar.selectbox(
    "Predefined Kinematics",
    list(presets.keys()),
    index=0,
)

param1 = st.sidebar.number_input(
    "Start Position  \n[(User Units)]",
    value=0.0,
    step=10.0,
    format="%.3f",
    icon=":material/line_start_circle:",
)

param2 = st.sidebar.number_input(
    "End Position  \n[(User Units)]",
    value=50.0,
    step=10.0,
    format="%.3f",
    icon=":material/line_end_circle:",
)

param3 = st.sidebar.number_input(
    "Speed  \n[(User Units)/sec]",
    value=presets[axis_choice]["speed"],
    step=10.0,
    format="%.3f",
    icon=":material/speed:",
)

param4 = st.sidebar.number_input(
    "Acc & Dec  \n[(User Units)/sec^2]",
    value=presets[axis_choice]["acc"],
    step=10.0,
    format="%.3f",
    icon=":material/motion_blur:",
)

param6 = st.sidebar.number_input(
    "Jerk  \n[(User Units)/sec^3]",
    value=presets[axis_choice]["jerk"],
    step=10.0,
    format="%.3f",
    icon=":material/fast_forward:",
)

# Generate and Display Plot
if st.button("Generate Plot"):
    figs, time_total, acc_rms = generate_plot2(param1, param2, param3, param4, param6)
    st.markdown(
        f"### Time To Perform The Defined Profile: `{time_total:.4f}` seconds"
    )
    st.plotly_chart(figs[0], use_container_width=True)
    st.plotly_chart(figs[1], use_container_width=True)
    st.plotly_chart(figs[2], use_container_width=True)
    st.info(f"RMS Acceleration (proxy for RMS current): {acc_rms:.3f} (User Units/sec²)")











