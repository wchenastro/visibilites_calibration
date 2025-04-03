import numpy as np

from scipy.interpolate import make_interp_spline


def make_sinusoidal_waveform(frequency, duration, sample_rate, batch,
        noise=None):
    """Make a sinusoidal waveform with the given parameters."""

    time_tick = np.linspace(
            0, duration, int(duration * sample_rate), endpoint=False)

    time_tick_group = np.tile(time_tick, (batch, 1))

    if noise:
        noise = np.random.normal(noise[0], noise[1], (batch, len(time_tick)))
        # noise = np.random.normal(noise[0], noise[1], len(time_tick))
    else:
        noise = np.zeros((batch, len(time_tick)))

    if hasattr(frequency, "__len__"):
        waveform = np.zeros((batch, len(time_tick)))
        for i, freq in enumerate(frequency):
            waveform[i] = np.sin(2 * np.pi * freq * time_tick_group[i]) + noise[i]
    else:
        waveform = np.sin(2 * np.pi * frequency * time_tick_group) + noise

    # waveform = np.sin(2 * np.pi * frequency * time_tick_group) + noise

    return waveform



def generate_complex_samples(shape, scale=1):
    '''
    Generate complex samples in given shape

    arguments:
        shape of the desired array of samples

    returns:
        uncorrelated samples in given shape
    '''
    shape[-1] = shape[-1] * 2
    uncorrelated_samples = np.random.normal(
        0, np.sqrt(2)/2 * scale, size=shape).astype("float64").view("complex128")
    return uncorrelated_samples;

def generate_random_curve(length, segment_len = 20, mean = [0], scale = 1):

    original_x = np.arange(segment_len)
    original_y_list = np.random.normal(
            np.array(mean)[:, np.newaxis], scale, [len(mean), segment_len])
    x_interp = np.linspace(original_x[0], original_x[-1], length)
    y_interp_list = []
    for index, original_y in enumerate(original_y_list):
        spline_interpolater = make_interp_spline(original_x, original_y, k=3)
        y_interp_list.append(spline_interpolater(x_interp))
    return np.array(y_interp_list)
