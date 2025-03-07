import mne

def badChanFixer(raw, subIdx, run):
    if subIdx == 12:
        bads = ['MEG 004']            

    raw.info['bads'] = bads
    raw.load_data()
    # Interpolate bad channels
    raw.interpolate_bads(reset_bads=True)
    return raw


def icaFixer(raw, ica, subIdx, run):
    if subIdx == 12:
        if run == 1:
            badComps = [0, 1, 19]
        elif run == 2:
            badComps = [0, 3, 7]
        elif run == 3:
            badComps = [0, 3, 7]
        elif run == 4:
            badComps = [0, 2, 9]
        elif run == 5:
            badComps = [0, 1, 7]
        elif run == 6:
            badComps = [0, 2, 8]
        elif run == 7:
            badComps = [0, 3, 14]
        elif run == 8:
            badComps = [0, 3, 9]
        elif run == 9:
            badComps = [0, 2, 9]
        elif run == 10:
            badComps = [0, 2, 7]
        elif run == 11:
            badComps = [0, 1, 10]

    ica.exclude = badComps
    raw.load_data()
    ica.apply(raw)

    return raw
