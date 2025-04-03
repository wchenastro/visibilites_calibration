
import numpy as np

def delay_signal(waveform_group, delay_group, observe_frequency, sign = 1, noise = 0):
    """Delay the given waveform by the given delay."""
    delayed_waveform_group = []
    fft_size = len(waveform_group[0])
    waveform_freq = np.fft.fftfreq(fft_size, 1/observe_frequency)
    if type(delay_group) == type(0) and delay_group == 0:
        delay_group = np.zeros(len(waveform_group))
    if type(noise) == type(0) and noise == 0:
        noise = np.ones(len(waveform_group))
    for i, waveform in enumerate(waveform_group):
        waveform_ffted = np.fft.fft(waveform)
        phase_shifter = np.exp(
                -1j * 2 * np.pi * sign * delay_group[i] * waveform_freq)
        delayed_waveform_ffted = waveform_ffted * phase_shifter * noise[i]
        delayed_waveform = np.fft.ifft(delayed_waveform_ffted)
        delayed_waveform_group.append(delayed_waveform)

    return np.array(delayed_waveform_group)
