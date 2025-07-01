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
        # This function gets two positions (pos0, pos1) and compute the movement time between those positions.
        # The arrival to destination position should be in speed =0, velocity =0.
        # Axis nominal kinematics is being used. pre-defined settling time [ms] is being added.
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
        # print(f' Axis {self.name}, Travel of {np.abs(pos1-pos0)}{self.units}, M&S duration {(trajectory.duration + settling_time):0.4f} [s]')
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
        otg = Ruckig(1,0.001)
        #print('\t'.join(['t'] + [str(i) for i in range(otg.degrees_of_freedom)]))
        # Generate the trajectory within the control loop
        first_output, out_list = None, []
        res = Result.Working
        while res == Result.Working:
            res = otg.update(inp, out)
            #print('\t'.join([f'{out.time:0.3f}'] + [f'{p:0.3f}' for p in out.new_position]))
            time.append(out.time)
            position.append(out.new_position)
            speed.append(out.new_velocity)
            acc.append(out.new_acceleration)
            out_list.append(copy(out))
            out.pass_to_input(inp)
    
            if not first_output:
                first_output = copy(out)
        return time, position,speed,acc

def generate_plot(start, finish, vel, acc, dec,jerk):
    axis1 = Axis("axis1",vel,acc,dec,jerk,'mm')
    t1 = axis1.time_to_perform(start, finish)
    time, position,speed,acc = axis1.online_trajectory(start, finish)
    fig, ax = plt.subplots(nrows=3, ncols=1,figsize=(24,20),dpi=100) # Optional: set figure size
    fig.subplots_adjust(hspace=0.4)
    
    ax[0].plot(time,position,label="Position")
    ax[0].legend()
    ax[0].grid()
    ax[0].set_title("Position")
    ax[0].set_xlabel("Time [sec]")
    ax[0].set_ylabel("[mm]")
    
    ax[1].plot(time,speed,label="Velocity")
    ax[1].legend()
    ax[1].grid()
    ax[1].set_title("Speed")
    ax[1].set_xlabel("Time [sec]")
    ax[1].set_ylabel("[mm/sec]") 

    ax[2].plot(time,speed,label="Acceleration")
    ax[2].legend()
    ax[2].grid()
    ax[2].set_title("Acceleration")
    ax[2].set_xlabel("Time [sec]")
    ax[2].set_ylabel("[mm/sec2]")           
              
    return(t1,fig)

def generate_plot2(start, finish, vel, acc, jerk):
    axis1 = Axis("axis1",vel,acc,acc,jerk,'mm')
    t1 = axis1.time_to_perform(start, finish)
    time, position,speed,acc = axis1.online_trajectory(start, finish)
    # Flatten Y data from [[x]] to [x]
    position = [p[0] for p in position]
    speed = [s[0] for s in speed]
    acc = [a[0] for a in acc]
    
    # Plot 1
    fig1 = go.Figure()
    fig1.add_trace(go.Scatter(x=time, y=position,mode="lines",hovertemplate="Time: %{x:.3f} s<br>Position: %{y:.2f} uut"))
    fig1.update_layout(title="Plot of Position", xaxis_title="Time [sec]", yaxis_title="(User Units)", hovermode="x unified")
    # Plot 2
    fig2 = go.Figure()
    fig2.add_trace(go.Scatter(x=time, y=speed,mode='lines',hovertemplate="Time: %{x:.3f} s<br>Velocity: %{y:.2f} uut/sec"))
    fig2.update_layout(title="Plot of Velocity", xaxis_title="Time [sec]", yaxis_title="(User Units)/sec", hovermode="x unified")
    # Plot 3
    fig3 = go.Figure()
    fig3.add_trace(go.Scatter(x=time, y=acc,mode='lines',hovertemplate="Time: %{x:.3f} s<br>Acceleration: %{y:.2f} uut/sec^2"))
    fig3.update_layout(title="Plot of Acceleration", xaxis_title="Time [sec]", yaxis_title="(User Units)/sec^2", hovermode="x unified")
    
         
              
    return (fig1, fig2, fig3), t1
                      



# Title
st.title("Interactive Motion Profile Generator")

# Sidebar Inputs
param1 = st.sidebar.number_input("Start Position  \n[(User Units)]", value=0.0,step=10.0,format="%.3f",icon=':material/line_start_circle:')
param2 = st.sidebar.number_input("End Position  \n[(User Units)]", value=50.0,step=10.0,format="%.3f",icon=':material/line_end_circle:')
param3 = st.sidebar.number_input("Speed  \n[(User Units)/sec]", value=1200.0,step=10.0,format="%.3f",icon=':material/speed:')
param4 = st.sidebar.number_input("Acc & Dec  \n[(User Units)/sec^$2$]",value=24000.0,step=10.0,format="%.3f",icon=':material/motion_blur:')
param6 = st.sidebar.number_input("Jerk  \n[(User Units)/sec^$3$]", value=600000.0,step=10.0,format="%.3f",icon=':material/fast_forward:')

# Generate and Display Plot
if st.button("Generate Plot"):
    #st.pyplot(fig)
    figs,time = generate_plot2(param1, param2, param3, param4, param6)
    st.markdown(f"### Time To Perform The Defined Profile: `{time:.2f}` seconds")
    st.plotly_chart(figs[0], use_container_width=True)
    st.plotly_chart(figs[1], use_container_width=True)
    st.plotly_chart(figs[2], use_container_width=True)
    
    
    
#fig=generate_plot(0,50,1200,24000,24000,600000)
