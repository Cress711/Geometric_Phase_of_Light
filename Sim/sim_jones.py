import numpy as np
import matplotlib.pyplot as plt

def jones_RCP():
    return (1 / np.sqrt(2)) * np.array([1.0, 1.0j], dtype=complex)


def jones_LCP():
    return (1 / np.sqrt(2)) * np.array([1.0, -1.0j], dtype=complex)

def R(theta):
    return np.array([
        [np.cos(theta), -np.sin(theta)],
        [np.sin(theta),  np.cos(theta)]
    ], dtype=complex)


def waveplate(theta, delta):
    return R(theta) @ np.array([
        [1.0, 0.0],
        [0.0, np.exp(1.0j * delta)]
    ], dtype=complex) @ R(-theta)


def QWP_matrix(theta):
    return waveplate(theta, np.pi / 2)


def HWP_matrix(theta):
    return waveplate(theta, np.pi)

def stokes_vector(E):
    Ex, Ey = E[0], E[1]

    S0 = np.abs(Ex)**2 + np.abs(Ey)**2
    S1 = np.abs(Ex)**2 - np.abs(Ey)**2
    S2 = 2 * np.real(np.conj(Ex) * Ey)
    S3 = 2 * np.imag(np.conj(Ex) * Ey)

    return np.array([S1, S2, S3], dtype=float) / S0

def QWP_trajectory(psi_in, theta=0.0, steps=300):
    traj = []

    for delta in np.linspace(0, np.pi / 2, steps + 1)[1:]:
        J = waveplate(theta, delta)
        psi = J @ psi_in
        traj.append(stokes_vector(psi))

    psi_out = QWP_matrix(theta) @ psi_in
    return np.array(traj), psi_out


def HWP_endpoint(psi_in, theta):
    return HWP_matrix(theta) @ psi_in


def equator_angle(S):
    return np.arctan2(S[1], S[0])


def wrap_to_pi(angle):
    return (angle + np.pi) % (2 * np.pi) - np.pi


def equator_arc(S_in, S_out, steps=300):
    lambda_start = equator_angle(S_in)
    lambda_end = equator_angle(S_out)

    delta = wrap_to_pi(lambda_end - lambda_start)

    if abs(delta) < 1e-10:
        delta = 2 * np.pi

    if np.isclose(abs(delta), np.pi, atol=1e-12):
        delta = np.pi

    lambdas = lambda_start + np.linspace(0, delta, steps + 1)[1:]

    traj = np.column_stack([
        np.cos(lambdas),
        np.sin(lambdas),
        np.zeros_like(lambdas)
    ])

    return traj


def qwp_theta_to_return_to_RCP(S_on_equator):
    lam = equator_angle(S_on_equator)

    return 0.5 * (lam - np.pi / 2)


def simulate(steps=300):
    theta_qwp1 = 0.0
    theta_hwp = np.pi / 8

    psi = jones_RCP()
    traj = [stokes_vector(psi)]

    t, psi = QWP_trajectory(psi, theta=theta_qwp1, steps=steps)
    traj.extend(t)

    S_before_HWP = stokes_vector(psi)

    psi_after_HWP = HWP_endpoint(psi, theta=theta_hwp)
    S_after_HWP = stokes_vector(psi_after_HWP)

    t = equator_arc(S_before_HWP, S_after_HWP, steps=steps)
    traj.extend(t)

    psi = psi_after_HWP

    theta_qwp2 = qwp_theta_to_return_to_RCP(stokes_vector(psi))

    t, psi = QWP_trajectory(psi, theta=theta_qwp2, steps=steps)
    traj.extend(t)
    return np.array(traj)


def RCP_stokes():
    return np.array([0.0, 0.0, 1.0])


def LCP_stokes():
    return np.array([0.0, 0.0, -1.0])


def plot_poincare():
    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111, projection="3d")

    u = np.linspace(0, 2 * np.pi, 100)
    v = np.linspace(0, np.pi, 100)

    x = np.outer(np.cos(u), np.sin(v))
    y = np.outer(np.sin(u), np.sin(v))
    z = np.outer(np.ones_like(u), np.cos(v))

    ax.plot_surface(x, y, z, alpha=0.12, color="gray")

    ax.set_xlabel("S1")
    ax.set_ylabel("S2")
    ax.set_zlabel("S3")

    ax.set_xlim([-1, 1])
    ax.set_ylim([-1, 1])
    ax.set_zlim([-1, 1])
    ax.set_box_aspect([1, 1, 1])

    return fig, ax



steps = 300
traj = simulate(steps=steps)

fig, ax = plot_poincare()

ax.plot(
    traj[:, 0],
    traj[:, 1],
    traj[:, 2],
    label="Jones: QWP(0) -> HWP(pi/8) -> QWP"
)

ax.scatter(*RCP_stokes(), color="blue", s=60, label="RCP")
ax.scatter(*LCP_stokes(), color="red", s=60, label="LCP")

ax.legend()
plt.title("Trajektoria liczona z macierzy Jonesa")
plt.show()
