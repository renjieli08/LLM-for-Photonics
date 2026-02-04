#Code written by GPT4 to simulate PCSEL. Authors: Hai Huang, Renjie Li, November 2025
import meep as mp
import numpy as np
import matplotlib.pyplot as plt
#parameter
a = 1 #zoom in rate of 1 μm.
fcen = 1/(0.98*a)
df = 0.1*fcen


etch = mp.Medium(index=1)
dpml = 0.98/2*a
pml_layers = [mp.PML(dpml)]

lattice = 0.298*a

    
r = 0.20*lattice
z_span = a
n_side = 30
edge = 1.0*a

lattice_o = 0.290*a
r_o = 0.33*lattice_o
z_deep = 1.0*a
n_i = 25
n_o =60


resolution = 1/0.015   # pixels/μm

t_nclad = 1.5*a+dpml
t_active = 0.216*a
t_pclad = 0.36*a
t_gaas = 0.1*a
t_air = 0.5*a+dpml

fill = 0.15

sx = lattice_o*(n_side+n_o+1)
sy = lattice_o*(n_side+n_o+1)
sz = t_nclad+t_active+t_gaas+t_pclad+t_air
# sz = 0
cell = mp.Vector3(sx+2*dpml,sy+2*dpml,sz)
k_point = mp.Vector3(0,0,0)


# Generate edge
groove_shift1 = (n_side+1)/2*lattice*np.array(((-1,-1),(-1,1),(1,1),
                                                                         (1,-1),(-1,-1)))
groove_shift2 = ((n_side+1)/2*(lattice)+2*edge)*np.array(((-1,-1),(1,-1),(1,1),
                                                                         (-1,1),(-1,-1)))


#simulation
geometry = [mp.Block(size=mp.Vector3(mp.inf,mp.inf,t_nclad),material=mp.Medium(index=3.2744),center=mp.Vector3(z=-sz/2+t_nclad/2)),
        mp.Block(size=mp.Vector3(mp.inf,mp.inf,t_active),material=mp.Medium(index=3.4),center=mp.Vector3(z=-sz/2+t_nclad+t_active/2)),
        mp.Block(size=mp.Vector3(mp.inf,mp.inf,t_pclad),material=mp.Medium(index=3.2744),center=mp.Vector3(z=-sz/2+t_nclad+t_active+t_pclad/2)),
        mp.Block(size=mp.Vector3(mp.inf,mp.inf,t_gaas),material=mp.Medium(index=3.4824),center=mp.Vector3(z=-sz/2+t_nclad+t_active+t_pclad+t_gaas/2)),
        ]
n_x = n_side
n_y = n_side

for k in np.arange(0, n_x, 1):
    for i in np.arange(0, n_y,1):
        geometry.append(mp.Cylinder(r, center = mp.Vector3(x = -(n_side-1)*lattice/2 + k*lattice, y = -(n_side-1)*lattice/2 + i*lattice , z=-sz/2+t_nclad+t_active+(t_pclad+t_gaas)/2), height=t_pclad+t_gaas, material = etch))

# for k in np.arange(0, n_i+n_o, 1):
#     for i in np.arange(0, n_i+n_o,1):
#         x_p = -(n_i+n_o-1)*lattice_o/2 + k*lattice_o
#         y_p = -(n_i+n_o-1)*lattice_o/2 + i*lattice_o
#         if (np.abs(x_p) > np.abs(-(n_side-1)*lattice/2) or np.abs(y_p) > np.abs(-(n_side-1)*lattice/2)):
#                 geometry.append(mp.Cylinder(r_o, center = mp.Vector3(x = x_p, y = y_p , z=-sz/2+t_nclad+t_active+t_pclad+t_gaas-z_deep/2), height=z_deep, material = etch))


# create clinder
n_y = int((n_side+n_o+1)/2)
print(n_y)
n_x = 2*n_y-8
for j in np.arange(-n_x/2, n_x/2+1,1): 
    x_p = j*lattice_o
    y_p = 0
    if (np.abs(x_p) > np.abs(-(n_side)*lattice/2) or np.abs(y_p) > np.abs(-(n_side)*lattice/2)):
        geometry.append(mp.Cylinder(r_o, center = mp.Vector3(x = x_p, y = y_p ,z=-sz/2+t_nclad+t_active+t_pclad+t_gaas-z_deep/2), height = z_deep))    
for i in range(1,n_y+1,1):
    if n_x%2 == 0:
        n_x = n_x-1
    else:
        n_x = n_x+1
    for j in np.arange(-n_x/2,n_x/2+1,1):
        x_p = j*lattice_o
        y_p= i*lattice_o*np.sqrt(3)/2
        if (np.abs(x_p) > np.abs(-(n_side)*lattice/2) and np.abs(y_p) < np.abs(-(n_side)*lattice/2)):
            geometry.append(mp.Cylinder(r_o, center = mp.Vector3(x = x_p, y = -y_p ,z=-sz/2+t_nclad+t_active+t_pclad+t_gaas-z_deep/2), height = z_deep))
            geometry.append(mp.Cylinder(r_o, center = mp.Vector3(x = x_p, y = y_p ,z=-sz/2+t_nclad+t_active+t_pclad+t_gaas-z_deep/2), height = z_deep))
        elif (np.abs(x_p) < np.abs(-(n_side)*lattice/2) and np.abs(y_p) > np.abs(-(n_side)*lattice/2)):
            geometry.append(mp.Cylinder(r_o, center = mp.Vector3(x = x_p, y = -y_p ,z=-sz/2+t_nclad+t_active+t_pclad+t_gaas-z_deep/2), height = z_deep))
            geometry.append(mp.Cylinder(r_o, center = mp.Vector3(x = x_p, y = y_p ,z=-sz/2+t_nclad+t_active+t_pclad+t_gaas-z_deep/2), height = z_deep))


vertices = [mp.Vector3(*_,z=-sz/2+t_nclad+t_active) for _ in groove_shift1.tolist()+groove_shift2.tolist()]
geo2 = [mp.Prism(vertices,height=(mp.inf),material=mp.Medium(index=1)),]
# geometry = geometry+geo2

sources = [mp.Source(src=mp.GaussianSource(fcen, fwidth=df),
                    component=mp.Hz,
                    center=mp.Vector3(x=0,z=-sz/2+t_nclad+t_active/2))]

sim = mp.Simulation(resolution=resolution,
                    cell_size=cell,
                    geometry=geometry,
                    sources=sources,
                    boundary_layers=pml_layers,
                    k_point=k_point)



resonance_top = mp.FluxRegion(center=mp.Vector3(0.3*n_side*lattice,0.3*n_side*lattice,-sz/2+t_nclad+t_active+t_pclad+t_gaas+0.4*a),size=mp.Vector3(0.6*n_side*lattice,0.6*n_side*lattice),direction=mp.Z)
resonance_side = mp.FluxRegion(center=mp.Vector3(0.65*n_side*lattice,0,0),size=mp.Vector3(0,n_side*lattice,sz-2*dpml),direction=mp.X)
pt = mp.Vector3(z=-sz/2+t_nclad+t_active/2)


# run sim
import sys

current = sys.stdout
f = open(f'./PCSEL-spc-{n_side}p.log', 'w+')
sys.stdout = f
sim.run(until_after_sources=100 / fcen)

# add DFT monitor
resonance_z = sim.add_flux(fcen, df*2, 101, resonance_top)
resonance_x = sim.add_flux(fcen, df*2, 101, resonance_side)
# nearfield_box = sim.add_near2far(fcen, df, 101,
#                                  mp.Near2FarRegion(center=mp.Vector3(z=-sz/2+t_nclad+t_active+t_pclad+t_gaas+0.4*a),
#                                                    size=mp.Vector3(sx,sy,0)),
#                                 mp.Near2FarRegion(center=mp.Vector3(z=-sz/2+t_nclad+t_active+t_pclad+t_gaas+0.4*a/2,x=-sx/2),
#                                                    size=mp.Vector3(0,sy,0.4*a),weight=-1),
#                                 mp.Near2FarRegion(center=mp.Vector3(z=-sz/2+t_nclad+t_active+t_pclad+t_gaas+0.4*a/2,x=sx/2),
#                                                    size=mp.Vector3(0,sy,0.4*a)),
#                                 mp.Near2FarRegion(center=mp.Vector3(z=-sz/2+t_nclad+t_active+t_pclad+t_gaas+0.4*a/2,y=-sy/2),
#                                                    size=mp.Vector3(sx,0,0.4*a),weight=-1),
#                                 mp.Near2FarRegion(center=mp.Vector3(z=-sz/2+t_nclad+t_active+t_pclad+t_gaas+0.4*a/2,y=sy/2),
#                                                    size=mp.Vector3(sx,0,0.4*a))
#                                 )

for _ in range(30):
    sim.run(mp.Harminv(mp.Hz, pt, fcen, 2*df),        
            until=49 / fcen)
    sim.run(mp.in_volume(mp.Volume(center=mp.Vector3(x=sx/4,y=sy/4,z=-sz/2+t_nclad+t_active/2),size=mp.Vector3(sx/2,sy/2)), 
                      mp.to_appended(f"Hz{_}", mp.at_every(1 / fcen / 20, mp.output_hfield_z))),
            until=1 / fcen)

# data save

z_flux = mp.get_fluxes(resonance_z)
x_flux = mp.get_fluxes(resonance_x)
flux_freqs = mp.get_flux_freqs(resonance_z)
wl = []
Ts_z = []
Ts_x = []
for i in range(len(flux_freqs)):
    wl = np.append(wl, 1 / flux_freqs[i])
    Ts_z = np.append(Ts_z, z_flux[i])
    Ts_x = np.append(Ts_x, x_flux[i])

import pickle
with open("./Ts_z_x.pickle", "wb") as f:
    pickle.dump([wl,Ts_z,Ts_x], f)

plt.plot(wl,Ts_z,label='z')
plt.plot(wl,Ts_x,label='x')
plt.xlabel("wavelength (μm)")
plt.legend()
plt.savefig('./Ts.png')

# near field Pz
nf = []
i = 0
for _ in resonance_z.freq:
    nf.append(sim.get_dft_array(resonance_z,mp.Ex,i)*sim.get_dft_array(resonance_z,mp.Hy,i)-sim.get_dft_array(resonance_z,mp.Ey,i)*sim.get_dft_array(resonance_z,mp.Hx,i))
    i += 1
with open("./near_field.pickle", "wb") as f:
    pickle.dump(nf, f)

# far field
# half side length of far-field square box
# r = 1000*a
# # resolution of far fields (points/μm)
# res_ff = 0.5
# ff = sim.get_farfields(nearfield_box,res_ff,center=mp.Vector3(z=r),size=mp.Vector3(sx+2*r,sy+2*r))
# with open("./far_field.pickle", "wb") as f:
#     pickle.dump(ff, f)

sys.stdout = current
f.close()
