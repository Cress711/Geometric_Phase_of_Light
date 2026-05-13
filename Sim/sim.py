import numpy as np
import matplotlib.pyplot as plt

#bieguny
def jones_RCP():
    return (1/np.sqrt(2)) * np.array([1, -1j])

def jones_LCP():
    return (1/np.sqrt(2)) * np.array([1, 1j])

#stokes z definicji
def stokes_vector(E):
    Ex, Ey = E[0], E[1]
    S0 = np.abs(Ex)**2 + np.abs(Ey)**2
    S1 = np.abs(Ex)**2 - np.abs(Ey)**2
    S2 = 2*np.real(np.conj(Ex)*Ey)
    S3 = 2*np.imag(np.conj(Ex)*Ey)
    return np.array([S1, S2, S3]) / S0

#obrót tylko wokół osi bo na razie nic bardziej skomplikowanego nie robimy
#ma tylko przejść z góy na dół
def rot_ax(theta):
    return np.array([np.cos(2*theta), np.sin(2*theta), 0])

def rotate_on_sphere(S, axis, angle):
    #rotacja działa tylko dla osi jednostkowej
    axis = axis / np.linalg.norm(axis)
    return (
        S*np.cos(angle)
        + np.cross(axis, S)*np.sin(angle)
        + axis*np.dot(axis, S)*(1-np.cos(angle))
    ) #to jest wzór Rodriguesa na obrot wektorów w 3D

def trajectory(theta, steps=200):
    S0 = stokes_vector(jones_RCP()) #start
    axis = rot_ax(theta)

    traj = []
    for phi in np.linspace(0, np.pi, steps):
        S = rotate_on_sphere(S0, axis, phi)
        traj.append(S)

    return np.array(traj)


def plot_poincare():
    fig = plt.figure(figsize=(8,8))
    ax = fig.add_subplot(111, projection='3d')

    u = np.linspace(0, 2*np.pi, 100)
    v = np.linspace(0, np.pi, 100)
    x = np.outer(np.cos(u), np.sin(v))
    y = np.outer(np.sin(u), np.sin(v))
    z = np.outer(np.ones_like(u), np.cos(v))

    ax.plot_surface(x, y, z, alpha=0.1, color='gray')

    ax.set_xlabel('S1')
    ax.set_ylabel('S2')
    ax.set_zlabel('S3')

    return fig, ax

angles = [0, np.pi/8, np.pi/4, 3*np.pi/8] #[rad]

fig, ax = plot_poincare()

for theta in angles:
    traj = trajectory(theta)
    ax.plot(traj[:,0], traj[:,1], traj[:,2], label=f'γ={theta:.2f}')

S_R = stokes_vector(jones_RCP())
S_L = stokes_vector(jones_LCP())

ax.scatter(*S_R, color='blue', s=50, label='RCP')
ax.scatter(*S_L, color='red', s=50, label='LCP')

ax.legend()
plt.title("Trajektorie na sferze Poincare (RCP → LCP przez półfalówkę)")
plt.show()
