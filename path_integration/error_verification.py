import numpy as np
import matplotlib.pyplot as plt
"""Verification of the performance error modeling from Muller and Wehner, 1988"""

pi = np.pi
rad = np.linspace(-pi, pi)
deg = rad * 180 / pi
conv = 180 / pi
sinmax = np.amax(deg * (180 + deg) * (180 - deg))
k_unit = 1/sinmax
k_path = 90/sinmax  # normalize fsin to a max of 90 degrees for path integration
N = deg.shape[0]
eps = np.finfo(float).eps

fcos = lambda x: 1 - np.abs(x) / 90
fsin = lambda x, k: x * (180 + x) * (180 - x) * k

# sin/cos approximation comparison
cos_out = np.cos(rad)
fcos_out = fcos(deg)

sin_out = np.sin(rad)
fsin_out = fsin(deg, k_unit)

f, ax = plt.subplots(2,1)
plt.sca(ax[0])
ax[0].plot(rad, cos_out, label='True')
ax[0].plot(rad, fcos_out, label='Approx')
plt.title('Cosine vs approximation')
plt.sca(ax[1])
ax[1].plot(rad, sin_out, label='True')
ax[1].plot(rad, fsin_out, label='Approx')
plt.title('Sine vs approx')

# steps in each direction
r0 = 10
r1 = 5
nsteps = r0 + r1

# absolute directions
angle = np.zeros([deg.shape[0], nsteps])
angle[:, r0:] = np.stack([deg] * r1, axis=1)

L1 = np.zeros([N, nsteps])
L2 = L1.copy()
A1 = np.zeros([N, nsteps])
A2 = A1.copy()

# iteratively update angles and distance
for i in range(nsteps):
    # linear approx
    D1 = angle[:, i] - A1[:, i-1]  # delta is angle between absolute new direction and absolute old
    L1[:, i] = L1[:, i-1] + fcos(D1)
    A1[:, i] = A1[:, i-1] + D1 / (L1[:, i-1] + eps)

    # polynomial approx
    D2 = angle[:, i] - A2[:, i-1]
    L2[:, i] = L2[:, i-1] + fcos(D2)
    A2[:, i] = A2[:, i-1] + fsin(D2, k_path) / (L2[:, i - 1] + eps)

# flip angle to position by 180 degrees to get angle of return
A1_posn = A1[:, -1]
A1_home = A1_posn + 180
A2_posn = A2[:, -1]
A2_home = A2_posn + 180

# true path length and angles
angle /= conv
angle_trig = np.stack([np.cos(angle), np.sin(angle)], axis=2)
path = np.cumsum(angle_trig, axis=1)
dir_posn = np.arctan2(path[:,:,1], path[:,:,0])[:,-1]
dir_home = dir_posn * conv + 180
dist = np.linalg.norm(path, axis=2)

# length and return angle plots
f_est, ax_est = plt.subplots(2,1)
plt.sca(ax_est[0])
est_len1 = plt.plot(deg, L1[:, -1], label='simple estimate')
est_len2 = plt.plot(deg, L2[:, -1], label='improved estimate')
true_len = plt.plot(deg, dist[:, -1], label='true')
plt.title('Estimated vs true distance')
plt.legend()

plt.sca(ax_est[1])
true_angle = plt.plot(deg, dir_home, label='true')
est_angle1 = plt.plot(deg, A1_home, label='simple estimate')
est_angle2 = plt.plot(deg, A2_home, label='improved estimate')
plt.title('Estimated vs true return angles')
plt.legend()

# angle error plots
error1 = np.abs(A1_home - dir_home)
error2 = np.abs(A2_home - dir_home)
f_err, ax_err = plt.subplots()
plt.plot(deg, error1, label='simple error')
plt.plot(deg, error2, label='improved error')
plt.title('Error comparison')
plt.legend()

plt.show()
