import numpy as np

# periodic
def wind_periodic(t, A=0.001, period=2.0, phase=0.0,
                  drift_rate=0.0, noise_std=0.00005, **kwargs):
    """
    A: periodic amplitude (m/s^2)
    period: period in seconds
    drift_rate: optional slow drift (m/s^3)
    """
    sinusoid = A * np.sin(2 * np.pi * (t / period) + phase)
    drift = drift_rate * t
    noise = np.random.normal(0, noise_std)
    total = sinusoid + drift + noise
    return total


# linear
def wind_linear(t, slope=0.0001, noise_std=0.0005, **kwargs):
    ramp = slope * t
    noise = np.random.normal(0, noise_std)
    total = ramp + noise
    return total


# step
def wind_step(t, slope=0.0001, noise_std=0.0005, **kwargs):
    ramp = slope * t
    noise = np.random.normal(0, noise_std)

    if t < 10:
        return ramp + noise
    else:
        return 2 * ramp + noise


# polynomial_like
def wind_polynomial_like(t, A=0.001, period=20, phase=2.0,
                         drift_rate=0.0002, noise_std=0.0005, **kwargs):
    omega = 2 * np.pi / period
    sinusoid = A * (np.sin(omega * t + phase) - 1)
    drift = drift_rate * t
    noise = np.random.normal(0, noise_std)
    total = sinusoid + drift + noise
    return total


def get_wind_function(wind_type):
    wind_map = {
        "periodic": wind_periodic,
        "linear": wind_linear,
        "step": wind_step,
        "polynomial_like": wind_polynomial_like,
    }

    if wind_type not in wind_map:
        raise ValueError(
            f"Unknown wind_type: {wind_type}. "
            f"Available choices: {list(wind_map.keys())}"
        )

    return wind_map[wind_type]
