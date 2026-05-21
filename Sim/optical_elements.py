import numpy as np

#normalizacja sfery do 1
def _normalize(v):
    v = np.asarray(v, dtype=float)
    norm = np.linalg.norm(v)
    return v / norm

#liczenie osi obrotu według przyjętej konwencji układu współrzędnych
#(S1, S2, S3) -> (x, y, z)
#teoretycznie w ogólnym przypadku tu nie powinno być S3 = 0, tylko też funkcja trygonometryczna,
#ale akurat startujemy z biegunu i nigdy nie obracamy wokół osi S3, więc i tak się wyzeruje
def _retarder_axis(theta):
    return np.array([np.cos(2 * theta), np.sin(2 * theta), 0.0])

#obracanie "punktu" S (wektora Stokesa)
def _rotate_vector(S, axis, angle):
    S = np.asarray(S, dtype=float) #wektor na float
    axis = _normalize(axis) #powinno działać bez bo cała sfera i wszystkie obliczenia są do znormalizowane do 1
                            #ale no nie działa więc zostawić
    #wzór Rodrigueza
    return (
        S * np.cos(angle)
        + np.cross(axis, S) * np.sin(angle)
        + axis * np.dot(axis, S) * (1 - np.cos(angle))
    )


def _rotation_trajectory(S_in, axis, angle, steps):
    traj = []
    for phi in np.linspace(0, angle, steps + 1)[1:]:#steps odcinków nie od 0
        S = _rotate_vector(S_in, axis, phi)
        traj.append(S)

    return np.array(traj), traj[-1] #zwraca trajektorię i punkt końcowy

#to do drugiej QWP
def _equator_angle(S):
    return np.arctan2(S[1], S[0])

#może wypąść trajketoria 270* no to trzeba zmienić na najkrószą drogę, czyli 90* w drugą stronę
def _wrap_to_pi(angle):
    return (angle + np.pi) % (2 * np.pi) - np.pi


def _equator_arc(S_in, S_out, steps):
    lambda_start = _equator_angle(S_in)
    lambda_end = _equator_angle(S_out)
    delta = _wrap_to_pi(lambda_end - lambda_start)

    #numerycznie się to sra i trzeba na sztorc ustawić małe wartości jak są dziwne wartości teta
    #jak punkt początkowy i końcowy na równiku są takie same
    #to rysujemy pełny obieg po równiku a nie zerowy odcinek
    if abs(delta) < 1e-10:
        delta = 2 * np.pi

    #i to samo w półokręgu
    #to jest przy okazji przypadek największej zmaiany fazy geometrycznej
    #najdłuższa droga minus jakiś malutki margines
    #bo dla równo pi jest ten sam zwrot polaryazcji czyli faza geom = 0
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

#z innego Rodrigueza i obliczeń wektorowych wychodzi taki wzorek
#na kąt obrotu drugiej QWP żeby wrócić do RCP
def qwp_theta_to_return_to_RCP(S_on_equator):
    lam = _equator_angle(S_on_equator)
    return 0.5 * (lam - np.pi / 2)