# %%
from util.io import *
from util.similarlity import *
import util.plot as plt
from util.dtw import accelerated_dtw

boat_list = load("data/boat/pkl/AIS_TYP_ID_89.pkl")
print(len(boat_list))
traj1 = boat_list[2].get_points()
traj2 = boat_list[3].get_points()
print(hausdorff(traj1, traj2), hausdorff(traj2, traj1))
print(dtw_distance(traj1, traj2))
# print(accelerated_dtw(traj1, traj2, 'euclidean', 1)[0])

plt.figure(1)

plt.plot(traj1[:, 0], traj1[:, 1])
plt.plot(traj2[:, 0], traj2[:, 1])

plt.show()