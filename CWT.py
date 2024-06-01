import mne
import numpy as np
import pywt
import matplotlib.pyplot as plt

# Define the file paths
file_paths = [
    r'C:\Users\vaibh\Downloads\Neuro\EEG data\subject1_cleaned.fif',
]

# Function to perform CWT on an epoch
def apply_cwt(epoch_data, frequencies, sfreq):
    scales = pywt.scale2frequency('cmor', 1/frequencies) * sfreq
    cwt_out = np.array([pywt.cwt(signal, scales, 'cmor', sampling_period=1/sfreq)[0] for signal in epoch_data])
    return cwt_out

# Frequency range for CWT
frequencies = np.linspace(1, 50, 100)  # Example range, adjust as needed

# Process each file
for file_path in file_paths:
    # Load epochs
    epochs = mne.read_epochs(file_path, preload=True)
    sfreq = epochs.info['sfreq']  # Sampling frequency
    
    # Initialize a list to hold CWT results for all epochs
    all_epochs_cwt = []
    
    # Apply CWT on each epoch
    for epoch_data in epochs.get_data():
        # Apply CWT
        cwt_result = apply_cwt(epoch_data, frequencies, sfreq)
        all_epochs_cwt.append(cwt_result)
    
    # Optionally visualize CWT of the first epoch of the first channel
    plt.figure(figsize=(10, 4))
    plt.imshow(np.abs(all_epochs_cwt[0][0]), extent=[0, epoch_data.shape[1]/sfreq, frequencies[0], frequencies[-1]], aspect='auto', cmap='jet', origin='lower')
    plt.colorbar(label='Magnitude')
    plt.ylabel('Frequency (Hz)')
    plt.xlabel('Time (s)')
    plt.title(f'CWT of Epoch 1, Channel 1 from {file_path.split("//")[-1]}')
    plt.show()
    
    # Here you could also save the CWT results, analyze them further, or proceed with other operations.

    '''
# Apply AutoReject to automatically clean the epochs
ar = AutoReject(n_interpolate=[1, 2, 4], random_state=42, picks=mne.pick_types(epochs.info, eeg=True), n_jobs=-1)
epochs_clean = ar.fit_transform(epochs)

# Optional: Identify and drop epochs with low amplitude (indicating lack of signal)
min_peak_to_peak_amplitude = 5e-6
bad_epochs = [i for i, epoch in enumerate(epochs_clean) if epoch.max() - epoch.min() < min_peak_to_peak_amplitude]
epochs_clean.drop(bad_epochs, reason='LOW_AMPLITUDE')

# Optionally plot the cleaned epochs to visually inspect them
scalings = {'eeg': 50e-6}  # Define scaling for EEG channels
epochs_clean.plot(n_epochs=1, n_channels=10, scalings=scalings, title='Cleaned Epochs', block=True, label="auto")
'''

'''
def create_four_second_events(raw, duration=4.0):
    sfreq = raw.info['sfreq']  # Sampling frequency
    num_samples = len(raw.times)
    event_duration = int(sfreq * duration)  # Convert duration to samples
    events = []
    for start in range(0, num_samples, event_duration):
        if start // event_duration < len(wordorder):
            events.append([start, 0, item_mapping.get(wordorder[start // event_duration], 0)])
        else:
            break
    return np.array(events)

events = create_four_second_events(raw)
'''
