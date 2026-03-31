import nibabel as nb
import matplotlib.pyplot as plt


subjID = 4
if subjID == 1:
    # For subject 01
    pthToOrig = '/System/Volumes/Data/d/DATD/datd/MEG_MGS/01MGS/hi_res/Jason_anat+18+t1_mprage_sag.img'
    pthToWriteTo = '/System/Volumes/Data/d/DATD/datd/MEG_MGS/MEG_BIDS/sub-01/anat/sub-01_task-mgs_T1w.nii.gz'
elif subjID == 3:
    # For subject 03
    pthToOrig = '/System/Volumes/Data/d/DATD/datd/MEG_MGS/03MGS/12+t1mprage/Sangi+12+t1mprage.img'
    pthToWriteTo = '/System/Volumes/Data/d/DATD/datd/MEG_MGS/MEG_BIDS/sub-03/anat/sub-03_task-mgs_T1w.nii.gz'
elif subjID == 4:
    # For subject 04
    pthToOrig = '/System/Volumes/Data/d/DATD/datd/MEG_MGS/04MGS/mri/Structural.img'
    pthToWriteTo = '/System/Volumes/Data/d/DATD/datd/MEG_MGS/MEG_BIDS/sub-04/anat/sub-04_task-mgs_T1w.nii.gz'

img = nb.load(pthToOrig)
# print(img)

# Visualize the image
# plt.figure()
# plt.imshow(img.get_fdata()[:, 50, :], cmap='gray')
# plt.show()


print(img)
# nb.save(img, pthToWriteTo)