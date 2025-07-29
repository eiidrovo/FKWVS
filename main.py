import numpy as np
import pandas as pd
from scipy.integrate import simpson
import cmath
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.transforms import Affine2D
from PIL import Image   


def load_data(path,dictionary):
    data = open(path,'r')
    for line in data:
        line=line.strip().split()
        dictionary[line[0]]=float(line[1])

    data.close()

def near_Values(array, value):
    lista = np.sort(array)  # aseguramos que esté ordenada
    for i, num in enumerate(lista):
        if value == num:
            return [num]
        elif value < num:
            if i == 0:
                return [lista[0]]
            else:
                return [lista[i-1], lista[i]]
    return [lista[-1]]

def interpolator(dataset,betha_Value,B,T,x_Value):
    betas= [float(k) for k in dataset.keys()]
    near_betha_values=near_Values(betas,betha_Value)
    Coeff_values=np.zeros(len(near_betha_values))
    for i,bn in enumerate(near_betha_values):
        Table_to_use=dataset[f'{bn}']
        BT_value=B/T
        BT_values=np.array([col for col in Table_to_use.columns if isinstance(col,float)])
        near_bt_values=near_Values(BT_values,BT_value)
        
        C_values=np.zeros(len(near_bt_values))

        for j,near_bt in enumerate(near_bt_values):
            C_values[j]=np.interp(x_Value,Table_to_use['x'],Table_to_use[near_bt])
        Coeff_values[i]=np.interp(BT_value,near_bt_values,C_values)

    return(np.interp(x_Value,near_betha_values,Coeff_values))


vessel_img=Image.open("Vessel.png") 
added_mass_tables = pd.read_excel('Added_mass_tables.xlsx', sheet_name=None)
damping_tables=pd.read_excel('Damping_tables.xlsx', sheet_name=None)
vessel_strips=np.loadtxt('vessel_strips.eiis',skiprows=1,delimiter=',')


load_data('wave_properties.eiis',data:=dict())
load_data('vessel_properties.eiis',data)


mass = data["vessel_mass"]
vessel_speed = data["vessel_speed"]
angle=data['angle']
wave_length=data['wave_length']
wave_height=data['wave_height']
g=data['gravity']
rho=data['rho']
LCG=data['LCG']
J=data['J']
LOA=data['LOA']
D=data['D']
draft=data['T']
sensor1_position=data['Sensor1']
sensor2_position=data['Sensor2']
gamma=rho*g
omega_meet=np.sqrt((2*np.pi*g)/wave_length)-2*np.pi*vessel_speed*np.cos(np.deg2rad(angle))
K=2*np.pi/wave_length
a33=np.zeros(vessel_strips.shape[0])
b33=np.zeros(vessel_strips.shape[0])

C_=np.zeros(vessel_strips.shape[0])
A_=np.zeros(vessel_strips.shape[0])

for i, strip in enumerate(vessel_strips):
    xvalue_for_mass=omega_meet**2*strip[1]/(2*g)
    xvalue_for_damping=omega_meet**2/(2*g)

    C= interpolator(added_mass_tables,strip[3],strip[1],strip[2],xvalue_for_mass)
    A= interpolator(damping_tables,strip[3],strip[1],strip[2],xvalue_for_damping)
    C_[i]=C
    A_[i]=A
    a33[i]=(C*rho*np.pi*strip[1]**2)/8
    b33[i]=rho*g**2*A**2/(omega_meet**3)

A33=simpson(a33,vessel_strips[:,0])
B33=simpson(b33,vessel_strips[:,0])
A55=simpson(a33*vessel_strips[:,0]**2,vessel_strips[:,0])
B55=simpson(b33*vessel_strips[:,0]**2,vessel_strips[:,0])
A53=-simpson(a33*vessel_strips[:,0],vessel_strips[:,0])
B53=-simpson(b33*vessel_strips[:,0],vessel_strips[:,0])
Mwp=simpson(vessel_strips[:,1]*vessel_strips[:,0],vessel_strips[:,0])
Iwp=simpson(vessel_strips[:,1]*vessel_strips[:,0]**2,vessel_strips[:,0])
Awp=simpson(vessel_strips[:,1],vessel_strips[:,0])

F_function= np.zeros((a33.shape[0],2),dtype=np.complex64) #First Column f3, Second Column f5

for i, _ in enumerate(F_function):
    f3=wave_height*\
        np.exp(-K*vessel_strips[i,2])*\
        np.exp(1j*K*vessel_strips[i,0])*\
        (gamma*vessel_strips[i,1]-omega_meet**2*a33[i]+1j*omega_meet*b33[i])
    
    f5=-f3*vessel_strips[i,0]
    F_function[i,0]=f3
    F_function[i,1]=f5

F3=simpson(F_function[:,0],vessel_strips[:,0])
F5=simpson(F_function[:,1],vessel_strips[:,0])

D11=-(omega_meet**2)*(mass+A33)+1j*B33*omega_meet+gamma*Awp
D12=omega_meet**2*(mass*LCG-A53)+1j*B53*omega_meet-gamma*Mwp
D21=omega_meet**2*(mass*LCG-A53)+1j*B53*omega_meet-gamma*Mwp
D22=-(omega_meet**2)*(J+A55)+1j*B55*omega_meet+gamma*Iwp #FKING EXPRESSION KEEP ME GUESSING FOR 5 HRS

D_matrix=np.array([[D11, D12],
                  [D21, D22]])

F_Vector=np.array([[F3],
                  [F5]])

Amplitudes=np.dot(np.linalg.inv(D_matrix),F_Vector)

phase_displacement=np.angle(Amplitudes[0])
phase_rotation=np.angle(Amplitudes[1])
amplitude_displacement=np.abs(Amplitudes[0])
amplitude_rotation=np.abs(Amplitudes[1])


fps = 60
Cycles=10
frames = int(fps * omega_meet*2*np.pi * Cycles)

fig, ax = plt.subplots(figsize=(10, 6))
ax.set_xlim(-0.65*LOA, 0.65*LOA)
ax.set_ylim(-5*D, 5*D)
ax.grid(True)

img_array = np.array(vessel_img)
image_obj = ax.imshow(img_array, extent=[-LOA/2, LOA/2, 0,D], zorder=2, animated=True)
wave_profile, = ax.plot([], [], lw=2, color='blue')


x = np.linspace(-0.65*LOA,0.65*LOA, 200)

def init():
    wave_profile.set_data([], [])
    return wave_profile, image_obj
    
def update(frame):
    
    t = frame / fps
    z_offset=amplitude_displacement* np.cos(omega_meet * t+phase_displacement)-draft
    rotation_offset=np.rad2deg(amplitude_rotation)*np.sin(omega_meet*t+phase_rotation)
    trans = Affine2D().translate(0, z_offset)\
        .rotate_deg_around(0, 0, rotation_offset) 

    image_obj.set_transform(trans + ax.transData)

    wave = wave_height * np.cos(omega_meet * t +K * x)
    wave_profile.set_data(x, wave)
    
    return image_obj,wave_profile
ani = animation.FuncAnimation(
    fig, 
    update, 
    frames=frames,
    init_func=init,
    interval=1000/fps, 
    blit=True, 
    repeat=True
)


plt.title(f'Movement Heave and pitch, λ:{wave_length}')

plt.show()



