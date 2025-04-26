import numpy as np


def preproc_pupil(df):
    nTrials = df.shape[0]
    nSampsToInterp = 25

    for trl in range(nTrials):
        # Extract X, Y, Pupil and XDAT for the current trial
        thisTrlX = df['X'].iloc[trl]
        thisTrlY = df['Y'].iloc[trl]
        thisTrlPupil = df['Pupil'].iloc[trl].astype(float)
        thisTrlXDAT = df['XDAT'].iloc[trl]

        # Only focus on epochs that have fixation 1, 2, 3
        validXDATSamps = np.where((thisTrlXDAT == 1) | (thisTrlXDAT == 2) | (thisTrlXDAT == 3))[0]
        thisTrlX = thisTrlX[validXDATSamps]
        thisTrlY = thisTrlY[validXDATSamps]
        thisTrlPupil = thisTrlPupil[validXDATSamps]

        # Find samples in X and Y that have NaN values
        nanXY =np.where(np.isnan(thisTrlX) | np.isnan(thisTrlY))[0]

        # Interpolate 25 samples before and after the NaN samples
        for nanSample in nanXY:
            startSample = max(0, nanSample - nSampsToInterp)
            endSample = min(len(thisTrlX), nanSample + nSampsToInterp)

            # Interpolate Pupil values
            thisTrlPupil[startSample:endSample] = np.interp(
                np.arange(startSample, endSample),
                np.delete(np.arange(startSample, endSample), nanSample - startSample),
                np.delete(thisTrlPupil[startSample:endSample], nanSample - startSample)
            )


    return df