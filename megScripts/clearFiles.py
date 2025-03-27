import os, socket

dirToClear = 'decoding'
fNamesToClear = ['classif_indivTargets_alpha.pkl']
# fNamesToClear = ['classif_indivTargets_crossfrequency.pkl']
hostname = socket.gethostname()
if socket.gethostname() == 'zod':
    bidsRoot = '/System/Volumes/Data/d/DATD/datd/MEG_MGS/MEG_BIDS'
else:
    bidsRoot = '/scratch/mdd9787/meg_prf_greene/MEG_HPC'
subList = [1, 2, 3, 4, 5, 6, 7, 9, 10, 12, 13, 15, 17, 
            18, 19, 23, 24, 25, 26, 27, 28, 29, 31, 32]
for subjID in subList:
    subName = 'sub-%02d' % int(subjID)
    derivativesRoot = os.path.join(bidsRoot, 'derivatives', subName)
    megRoot = os.path.join(derivativesRoot, 'meg')
    classifHolderRoot = os.path.join(megRoot, dirToClear)
    for fName in fNamesToClear:
        fPath = os.path.join(classifHolderRoot, fName)
        if os.path.exists(fPath):
            os.remove(fPath)
            print('Removed ' + fPath)
        else:
            print(fPath + ' does not exist')
print('Done')
