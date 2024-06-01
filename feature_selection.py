import mne
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import numpy as np
from skimage.restoration import denoise_wavelet
import numpy as np
import seaborn as sns
import pywt






item_mapping = {"child": 0, "daughter": 0, "father": 0, "wife": 0, "four": 1, "three": 1, "ten": 1, "six": 1}


# Dictionary of EEG files and their corresponding event files
files = {
    r"C:\Users\vaibh\Downloads\Neuro\Events and Timings\sub-01_ses-EEG_task-inner_eeg (1).bdf": r"C:\Users\vaibh\Downloads\Neuro\Events and Timings\sub-01_ses-EEG_task-inner_events.txt",
    r"C:\Users\vaibh\Downloads\Neuro\Events and Timings\sub-05_ses-EEG_eeg.bdf": r"C:\Users\vaibh\Downloads\Neuro\Events and Timings\sub-05_ses-EEG_task-inner_events.txt",
    r"C:\Users\vaibh\Downloads\Neuro\Events and Timings\sub-02_ses-EEG_eeg.bdf": r"C:\Users\vaibh\Downloads\Neuro\Events and Timings\sub-02_ses-EEG_task-inner_events.txt",
    r"C:\Users\vaibh\Downloads\Neuro\Events and Timings\sub-03_ses-EEG_eeg.bdf": r"C:\Users\vaibh\Downloads\Neuro\Events and Timings\sub-03_ses-EEG_task-inner_events.txt",
}

# Function to process each EEG file
def process_eeg(eeg_file_path, events_file_path):
    # Load EEG data
    raw = mne.io.read_raw_bdf(eeg_file_path, preload=True)
    raw.drop_channels(['EXG7', 'EXG8'])  # Drop unwanted channels
    channels_of_interest = ['F3']
    raw.pick_channels(channels_of_interest)
    raw.filter(l_freq=0.5, h_freq=50., method='fir', fir_window='hamming', fir_design='firwin')
    # Notch filter to remove line noise
    notches = np.arange(50, 251, 50)
    raw.notch_filter(notches, picks='eeg', filter_length='auto', phase='zero-double', fir_design='firwin')

    # Collect word orders from the events file
    wordorder = []
    with open(events_file_path, "r") as x:
        for line in x:
            store = line.split()
            wordorder.append(store[2])
    wordorder = wordorder[1:]  # Remove header or initial unwanted line if necessary

    # Create and process events
    events = create_middle_two_second_events(raw, wordorder)
    epochs = mne.Epochs(raw, events, event_id=None, tmin=0, tmax=2.0, baseline=(None, None), preload=True)
    return epochs  # or any other data you wish to return or save

# Function to create events for the middle two seconds of each four-second segment, adjusted to use word order
def create_middle_two_second_events(raw, wordorder, duration=4.0):
    sfreq = raw.info['sfreq']
    num_samples = len(raw.times)
    event_duration_samples = int(sfreq * duration)
    middle_offset_samples = int(sfreq * (duration / 2.0 - 1))

    events = []
    for start in range(0, num_samples, event_duration_samples):
        if start // event_duration_samples < len(wordorder):
            middle_start = start + middle_offset_samples
            if middle_start < num_samples:
                event_id = item_mapping.get(wordorder[start // event_duration_samples], 0)
                events.append([middle_start, 0, event_id])
        else:
            break
    return np.array(events)

def make_epochs():  
    
    all_epochs_list = []
    all_labels = []
    # Iterate over the files dictionary and process each EEG file
    for eeg_path, events_path in files.items():
        epochs = process_eeg(eeg_path, events_path)
        wordorder = []
        with open(events_path, "r") as x:
            for line in x:
                if line.strip():  # skip empty lines
                    store = line.split()
                    wordorder.append(store[2])
        wordorder = wordorder[1:]  # Skip header or initial unwanted line if necessary
        
        # Append the epochs from this file to the list
        all_epochs_list.append(epochs)
        
        # Create labels for these epochs based on the wordorder
        file_labels = [item_mapping[word] for word in wordorder]
        all_labels.extend(file_labels)

    # Concatenate all epochs into a single Epochs object
    all_epochs = mne.concatenate_epochs(all_epochs_list)
    return all_epochs, all_labels

def denoise_emg_epochs(epochs):
    denoised_epochs = []

    # Iterate through each epoch
    for epoch in epochs:
        denoised_epoch = []

        # Iterate through each channel in the epoch
        for channel_data in epoch:
            # Apply the denoise_emg_signal function
            denoised_channel = denoise_emg_signal(channel_data)
            denoised_epoch.append(denoised_channel)
        
        # Stack the denoised channels back into an epoch
        denoised_epochs.append(np.stack(denoised_epoch, axis=0))
    
    # Stack all denoised epochs into a 3D numpy array to return
    return np.stack(denoised_epochs, axis=0)


def denoise_emg_signal(emg_signal):
    # Normalize the raw EMG data
    normalized_emg = emg_signal / np.amax(np.abs(emg_signal))
    
    # Apply wavelet denoising
    denoised_emg = denoise_wavelet(normalized_emg, method='BayesShrink', mode='soft',
                                   wavelet='sym8', wavelet_levels=5, rescale_sigma='True')
    
    return denoised_emg

def process_and_transform_epochs(epochs_data, labels):
    # Denoise the epochs
    denoised_epochs = denoise_emg_epochs(epochs_data)
    
    # Apply transformation to each denoised epoch
    transformed_epochs = np.array([denoise_emg_signal(epoch) for epoch in denoised_epochs])
    
    # The labels array is unchanged and can be returned alongside the transformed data
    return transformed_epochs, labels


epochs, labels = make_epochs()

transformed_epochs, updated_labels = process_and_transform_epochs(epochs, labels)
np.savetxt("save.txt", transformed_epochs[0])
