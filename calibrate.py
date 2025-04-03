#!/usr/bin/env python3

import numpy as np
from delay_calc import delay_signal
from gain_cal import stefcal
from signal_gen import generate_complex_samples
from signal_gen import generate_random_curve
from matplotlib import pyplot as plt


def calibrate():
    # observation parameters
    antenna_len = 5
    sample_len = int(1e4)
    channel_len = sample_len
    observe_frequency = 2.0e6

    # calibration parameters
    enable_instrument_noise = True
    enable_time_delay = True
    enable_phase_noise = True
    enable_baseline_fit = True
    phase_noise_scale = 5.0e-9
    phase_noise_trend = 7 # larger number mean more twists
    # how different between the means of the phase noise
    phase_noise_mean_offsets_scale = 1e-8
    instrument_noise_scale = 1e-3
    delay_and_scale = np.array([1, -1, 2, -3, 5]) * 1.0e-8

    # create the original signal
    source = generate_complex_samples([sample_len], scale = 1)

    phase_noise_mean = (np.random.uniform(0, np.pi*2, antenna_len)
                        * phase_noise_mean_offsets_scale)

    # create the phase curve through out the bandpass
    random_phase_curve = generate_random_curve(channel_len, phase_noise_trend,
            phase_noise_mean, phase_noise_scale)

    # create the fft frequencies
    fft_freq = np.fft.fftfreq(sample_len, 1/observe_frequency)

    # create the phase noise shifter from the random phase curve
    phase_noise = (np.exp(-1j * 2 * np.pi * random_phase_curve * fft_freq)
            if enable_phase_noise else 0)

    # create the instrument noise
    instrument_noise = (generate_complex_samples(
            [antenna_len, sample_len], scale = instrument_noise_scale)
            if enable_instrument_noise else 0)

    # fill the same model signal to each antenna
    source_array = np.tile(source, (antenna_len, 1))
    # apply the delay and noise into the model signal
    time_delay = delay_and_scale if enable_time_delay else 0
    source_signal = (delay_signal(source_array, time_delay, observe_frequency,
                1, noise = phase_noise) + instrument_noise)

    # channelize the signal
    channellized_signal = np.fft.fft(source_signal)
    channellized_model = np.fft.fft(source_array)

    # create the model and noisy (uncorrelated) visibilities
    """ row-wise outer product """
    model = np.einsum('ik,jk->ijk',
            channellized_model, channellized_model.conj())
    visibilities = np.einsum('ik,jk->ijk',
            channellized_signal, channellized_signal.conj())

    fig, axis = plt.subplots(nrows=2, ncols=2)
    # uncorrected visibilities
    ax0= axis[0, 0]
    phases = np.angle(visibilities[0, 4])
    phases_model = np.angle(model[0, 4])
    ax0.plot(np.fft.fftshift(fft_freq),
             np.unwrap(np.fft.fftshift(phases)), label="uncorrected")
    #  ax0.plot(np.fft.fftshift(fft_freq), np.fft.fftshift(phases_model), label="model")
    ax0.set_xlabel('Frequency')
    ax0.set_ylabel('Phase')
    ax0.set_title('Uncorrected')
    ax0.legend()

    # integrate
    integrate_factor = 10
    integrated_visibilities = integrate(visibilities, integrate_factor)
    integrated_model = integrate(model, integrate_factor)
    freqs = np.fft.fftshift(
            np.fft.fftfreq(int(channel_len/integrate_factor), 1/observe_frequency))
    phases = np.unwrap(np.fft.fftshift(np.angle(integrated_visibilities[0, 4])))
    phases_model = np.fft.fftshift(np.angle(integrated_model[0, 4]))
    # fit the slope of the integrated phase
    if enable_baseline_fit:
        conef = np.polyfit(freqs, phases, deg=1)
        fitted_phases = np.polyval(conef, freqs)
    else:
        fitted_phases = np.zeros_like(phases)

    ax1 = axis[0, 1]
    ax1.plot(freqs, phases, label='integrated')
    ax1.plot(freqs, fitted_phases, label='fit')
    ax1.set_xlabel('Frequency')
    ax1.set_ylabel('Phase')
    ax1.set_title('Integrated')
    ax1.legend()

    # flatten the integrated visibilities
    flattened_integrated_visibilities = (integrated_visibilities
            * np.exp(-1j * fitted_phases))
    #  flattened_phases = np.angle(flattened_integrated_visibilities[0,4])
    flattened_phases = phases - fitted_phases
    ax2 = axis[1, 0]
    ax2.plot(freqs,  flattened_phases, label='flattened')
    ax2.plot(freqs, phases_model, label='model')
    ax2.set_xlabel('Frequency')
    ax2.set_ylabel('Phase')
    ax2.set_title('Flattened' if enable_baseline_fit else "Not flattened")
    ax2.legend()

    # correct the visibilities
    corrected_visibilities = np.zeros_like(integrated_model)
    for channel_index in range(len(freqs)):
        gains = stefcal(
                flattened_integrated_visibilities[:, :, channel_index],
                integrated_model[:, :, channel_index], niter=100)
        per_antenna_gain = gains[0]
        correction_matrix  = np.outer(np.conj(per_antenna_gain), per_antenna_gain)
        flattened_integrated_visibilities[:, :, channel_index] *= (
                np.conj(flattened_integrated_visibilities[0, 0, channel_index])
                / np.abs(flattened_integrated_visibilities [0, 0, channel_index]))
        corrected_visibilities[:, :, channel_index] = (correction_matrix**(-1)
                * flattened_integrated_visibilities[:, :, channel_index])

    reference_frequency = 0.7e6
    reference_index = np.where(freqs == reference_frequency)[0].squeeze()

    ax3 = axis[1, 1]
    ax3.plot(freqs, np.angle(corrected_visibilities[0, 4]), label='corrected')
    ax3.plot(freqs, phases_model, label='model')
    ax3.set_xlabel('Frequency')
    ax3.set_ylabel('Phase')
    ax3.set_title('Corrected')
    ax3.legend()

    print("model visibilities")
    print(integrated_model[:, :, reference_index])
    print("flattened visibilities")
    print(flattened_integrated_visibilities[:, :, reference_index])
    print("corrected visibilities")
    print(corrected_visibilities[:, :, reference_index])
    plt.show()

def main():
    calibrate()

def integrate(data, factor=1):
    integrated_data = data.reshape(
            np.hstack([data.shape[:-1], [-1, factor]])
            ).mean(axis = len(data.shape))

    return integrated_data

if __name__ == '__main__':
    main()
