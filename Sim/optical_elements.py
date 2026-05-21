import numpy as np


def _normalize(v):
    v = np.asarray(v, dtype=float)
    norm = np.linalg.norm(v)

    if norm == 0:
        raise ValueError("Oś rotacji nie może być zerowa.")

    return v / norm


def _retarder_axis(theta):
    return np.array([
        np.cos(2 * theta),
        np.sin(2 * theta),
        0.0
    ])


def _rotate_vector(S, axis, angle):
    S = np.asarray(S, dtype=float)
    axis = _normalize(axis)

    return (
        S * np.cos(angle)
        + np.cross(axis, S) * np.sin(angle)
        + axis * np.dot(axis, S) * (1 - np.cos(angle))
    )


def _rotation_trajectory(S_in, axis, angle, steps):
    if steps < 2:
        raise ValueError("steps musi być >= 2.")

    traj = []

    for phi in np.linspace(0, angle, steps + 1)[1:]:
        S = _rotate_vector(S_in, axis, phi)
        traj.append(S)

    return np.array(traj), traj[-1]


def _equator_angle(S):
    return np.arctan2(S[1], S[0])


def _wrap_to_pi(angle):
    return (angle + np.pi) % (2 * np.pi) - np.pi


def _equator_arc(S_in, S_out, steps):
    if steps < 2:
        raise ValueError("steps musi być >= 2.")

    lambda_start = _equator_angle(S_in)
    lambda_end = _equator_angle(S_out)

    delta = _wrap_to_pi(lambda_end - lambda_start)

    # przypadek dokładnie półokręgu: wybieramy dodatni kierunek
    if np.isclose(abs(delta), np.pi, atol=1e-12):
        delta = np.pi

    lambdas = lambda_start + np.linspace(0, delta, steps + 1)[1:]

    traj = np.column_stack([
        np.cos(lambdas),
        np.sin(lambdas),
        np.zeros_like(lambdas)
    ])

    return traj, traj[-1]


def QWP(S_in, theta=0.0, steps=200):
    axis = _retarder_axis(theta)
    return _rotation_trajectory(S_in, axis, np.pi / 2, steps)


def HWP(S_in, theta=0.0, steps=200, path="physical"):
    axis = _retarder_axis(theta)
    S_out = _rotate_vector(S_in, axis, np.pi)

    if path == "equator":
        if abs(S_in[2]) < 1e-8 and abs(S_out[2]) < 1e-8:
            return _equator_arc(S_in, S_out, steps)

    return _rotation_trajectory(S_in, axis, np.pi, steps)


def qwp_theta_to_return_to_RCP(S_on_equator):
    lam = _equator_angle(S_on_equator)
    return 0.5 * (lam - np.pi / 2)