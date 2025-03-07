import numpy as np
import re, mne
from os import SEEK_CUR

def read_elp(fname: str) -> np.ndarray:
    """
    Reads an ELP (electrode location protocol) file and extracts fiducial and sensor data.
    
    Parameters:
    -----------
    fname: str
        Path to the ELP file.
        
    Returns:
    --------
    elpData: np.ndarray
        A numpy array containing the fiducials (nasion, left preauricular, right preauricular) and sensor data.
        
    """

    # This has been updated by Mrugank to work with the fiducials 
    with open(fname) as fid:
        file_str = fid.readlines()
    
    # Extract fiducials (lines starting with `%F`)
    fiducials = [
        re.findall(r"-?\d+\.\d+(?:[eE][-+]?\d+)?", line) 
        for line in file_str if line.startswith("%F")
    ]

    # Sometimes this fails if the thrid number is 0.0, so we need to add it manually
    for i in range(len(fiducials)):
        if len(fiducials[i]) == 2:
            fiducials[i].append("0.0")

    fiducials = np.array(fiducials, dtype=float)  # Convert to numpy array

    if len(fiducials) < 3:
        raise ValueError("Insufficient fiducials found in the file. Check `%F` entries.")
    
    # Extract sensor data (lines starting with `%N`)
    sensor_data = []
    for i, line in enumerate(file_str):
        if re.match(r"%N\s", line):
            if i + 1 < len(file_str):
                next_line = file_str[i + 1].strip()
                numbers = re.findall(r"-?\d+\.\d+(?:[eE][-+]?\d+)?", next_line) 
                sensor_data.append(numbers)
    
    # Flatten the list of sensor data and reshape it to 3 columns
    sensor_data = [item for sublist in sensor_data for item in sublist]
    sensor_data = [sensor_data[i:i + 3] for i in range(0, len(sensor_data), 3)]

    sensor_data = np.array(sensor_data, dtype=float)  # Convert to numpy array

    # Prepare the ELP data dictionary
    elpDict = {
        "nasion": fiducials[0],
        "lpa": fiducials[1],
        "rpa": fiducials[2],
        "points": sensor_data,
    }

    # Combine fiducials and sensor points into one array
    elpData = np.concatenate(
        (np.array([elpDict["nasion"], elpDict["lpa"], elpDict["rpa"]]), elpDict["points"]),
        axis=0,
    )
    
    return elpData


def read_hsp(fname: str) -> np.ndarray:
    """
    Reads an HSP (headshape) file and extracts the headshape points.
    
    Parameters:
    -----------
    fname: str
        Path to the HSP file.
        
    Returns:
    --------
    hspData: np.ndarray
        A numpy array containing the headshape points.
        
    """

    with open(fname) as fid:
        file_str = fid.readlines()

    # Extract headshape points (lines starting with `%`)
    headshape = []
    for i, line in enumerate(file_str):
        if re.match(r"//No", line):
            next_line = file_str[i + 1].strip()
            numPoints = next_line.split()[0]
            starterIdx = i + 2
 

    for i, line in enumerate(file_str):
        if i + 1 < len(file_str):
            if i >= starterIdx and i < starterIdx + int(numPoints):
                thPoint = re.findall(r"-?\d+\.\d+(?:[eE][-+]?\d+)?", line.strip())
                headshape.append(thPoint)

    headshape = np.array(headshape, dtype=float)  # Convert to numpy array

    return headshape

def extractEvents(raw):
    misc_channels = ['MISC 001', 'MISC 002', 'MISC 003', 'MISC 004', 'MISC 005', 'MISC 006']
    event_ids = [1, 2, 3, 4, 5, 6]  # Corresponding event IDs for each MISC channel
    # EPOCHS:
    #   1: Fixation
    #   2: Sample
    #   3: Delay
    #   4: Response
    #   5: Feedback
    #   6: ITI

    events_list = []
    for misc_channel, event_id in zip(misc_channels, event_ids):
        # Extract events from the given channel
        events = mne.find_events(raw, stim_channel=misc_channel, 
                                initial_event=False, min_duration=0.01, verbose=False)
        events[:, 2] = event_id  # Replace the event type with the corresponding event ID
        events_list.append(events)

    all_events = np.vstack(events_list)
    all_events = all_events[np.argsort(all_events[:, 0])] # Sort by time/sample

    return all_events