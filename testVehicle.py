import minecraftControl.vehicles as vh
import minecraftControl.minecraft as mc
import matplotlib.pyplot as plt
import numpy as np

layout = mc.smallLayout
x,y=  layout.start
vehicle = vh.car((-x,-1,y),controller=None)

plt.figure()

vehicle = vh.quadcopter((0.,0,1),controller=None)

mc.generalSystem(layout,vehicle)
# This is special to the quadcopter
#print(vehicle.v_traj)
plt.plot(vehicle.Time,np.array(vehicle.v_traj))
plt.show()
