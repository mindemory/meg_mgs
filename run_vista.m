%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% This script fits the smoothed, averaged functional surface giis from subject's pRF scan
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Attention attention attention -- all editors please read!
% This script operates on only one hemisphere, so a second script is needed to run in parallel.
% Therefore a corresponding script exists in the same parent directory. It is completely directory except all references to the hemisphere being used is switched.
% If ever you make an change to the script, please be sure all changes are made to the sister script as well. E.g., after your edits, make a copy and edit all references to hemisphere.
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Run script ('end' at very bottom)
function run_vista(fMRIPrepOUTDIR, RecID, basePATH, space, SmoothedOUTDIR, numRuns, hemi)

% Convert 'numRuns' input from integer to string
numRuns=str2double(numRuns);

% Ensure the following paths exist to call subsequent necessary functions
addpath(genpath('/scratch/cc99/CBI/pRFsurf_vista/gifti'));
addpath(genpath('/scratch/cc99/CBI/pRFsurf_vista/gridfitgpu'));
addpath(genpath('/scratch/cc99/CBI/pRFsurf_vista/preproc_mFiles'));
addpath(genpath('/scratch/cc99/CBI/pRFsurf_vista/pRFsurf_vista_fit'));
addpath(genpath('/scratch/cc99/CBI/pRFsurf_vista/vistasoft_ts'))

% addpath(genpath('/System/Volumes/Data/d/DATD/home/mrugank/pRF_greene_setup/toolboxes/vistasoft_ts'))
% Create output directory
% VistaDIR=sprintf('%s/pRFsurf_vista/pRFsurf_vista_fit/sub-%s',basePATH,RecID);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Average the fsnative smoothed.func.giis to create one per hemisphere
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% rootFuncs = '/d/DATD/datd/popeye_pRF/sub-MAM0606/ses-pRF/func_smoothed';

numRuns = 2;
% Create array per hemisphere with every func.gii.
% Use zero-padding with run numbers.
for rr = 1:numRuns
    disp(['Running run: ' num2str(rr, '%02d')])
    if rr < 10
        formatted_rr = sprintf('%02d', rr);
    else
        formatted_rr = num2str(rr);
    end


    % Create gifti object and add to array
    runL = sprintf('%s/sub-%s_ses-pRF_task-TASK_run-%s_hemi-L_space-%s–smoothed_bold.func.gii',SmoothedOUTDIR,RecID,formatted_rr,space);
    runR = sprintf('%s/sub-%s_ses-pRF_task-TASK_run-%s_hemi-R_space-%s–smoothed_bold.func.gii',SmoothedOUTDIR,RecID,formatted_rr,space);
    
    % runL = [rootFuncs filesep 'sub-MAM0606_ses-pRF_task-TASK_run-' formatted_rr '_hemi-L_space-fsnative_smoothed_bold.func.gii'];
    % runR = [rootFuncs filesep 'sub-MAM0606_ses-pRF_task-TASK_run-' formatted_rr '_hemi-R_space-fsnative_smoothed_bold.func.gii'];
    
    gL = gifti(runL);
    gR = gifti(runR);
    if rr == 1
        avgRuns_left = gL.cdata;
        avgRuns_right = gR.cdata;
    else
        avgRuns_left = cat(3, avgRuns_left, gL.cdata);
        avgRuns_right = cat(3, avgRuns_right, gR.cdata);
    end
end

% Divide array by number of runs to create an averaged file and save
% Average across runs
avgRuns_left = squeeze(mean(avgRuns_left, 3, 'omitnan')); 
avgRuns_right = squeeze(mean(avgRuns_right, 3, 'omitnan'));
avgLeft_gii = struct();
avgRight_gii = struct();
avgLeft_gii.data = avgRuns_left;
avgRight_gii.data = avgRuns_right;
% to-do: figure out a way to add header CortexLeft or CortexRight, do the
% workbench thing for both pre and post fitting files
% avgLeft_path = sprintf('%s/sub-%s_ses-pRF_hemi-L_Averaged-BOLD.mat',VistaDIR,RecID,space);
avgLeft_path = [rootFuncs filesep 'sub-MAM0606_ses-pRF_task-TASK_avg_hemi-L_space-fsnative_smoothed_bold.func.gii'];
save(avgLeft_path, 'avgLeft_gii');
avgRight_path = [rootFuncs filesep 'sub-MAM0606_ses-pRF_task-TASK_avg_hemi-R_space-fsnative_smoothed_bold.func.gii'];
save(avgRight_path, 'avgRight_gii');

% Announce section over
disp('Smoothed func.giis have been averaged into a single file per hemisphere. Starting model fit...')

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Convert surf files to volumetric space
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% % % % giftiload = sprintf('%s/LH_BOLDpRF.func.gii',SmoothedOUTDIR);
% % % % giftiobject = gifti(giftiload);
% % % % LH_BOLDpRF.func.gii = giftiobject.cdata;

% creating a fake folder for this subject that will be used to store nii
% files
fakeHolder = '/d/DATD/datd/popeye_pRF/fakeHolder';
subNewTemp = '/d/DATD/datd/popeye_pRF/WIC0326';
copyfile(fakeHolder, subNewTemp);

% Optional/debugging - confirm vertices (both files should have %%%, assuming fsnative)
numVerticesLeft = size(avgLeft_gii.data,1);
numVerticesRight = size(avgRight_gii.data,1);

% Get a volumetric placeholder as required by vista and save surface data to it
% niftiload = sprintf('/%s/%s/derivatives/sub-%s/ses-pRF/func/sub-%s_ses-pRF_task-TASK_run-01_space-T1w_desc-preproc_bold.nii.gz',basePATH,fMRIPrepOUTDIR,RecID,RecID);
niftiload = [subNewTemp '/RF1/WIC0326_RF1_vista/bar_seq_1_ss5.nii.gz'];
niftiobject = niftiRead(niftiload);
data = zeros(niftiobject.dim); % Initalize a data holder
data = reshape(data, prod(niftiobject.dim(1:3)), niftiobject.dim(4));
% Faking gii into nii
% data(1:numVerticesLeft, :) = avgLeft_gii.data;
% data(numVerticesLeft+1:numVerticesLeft+numVerticesRight, :) = avgRight_gii.data;

% For debugging comment out the last two lines and instead do
% This will only try fitting on first 1000 voxels which would be faster to
% test
data(1:1000, :) = avgLeft_gii.data(1:1000, :);

% Copy brain mask corresponding to volume placeholder, as required by vista. (Used as 'InPlane'.)
%BMorig = sprintf('%s/%s/derivatives/sub-%s/ses-pRF/func/sub-%s_ses-pRF_task-TASK_run-01_space-T1w_desc-brain_mask.nii.gz',basePATH,fMRIPrepOUTDIR,RecID,RecID,sesName);
%BMcopy = sprintf('%s/T1w-brain_mask.nii.gz',VistaDIR);
%copyfile(BMorig,BMcopy);
%Inplane = 'T1w-brain_mask.nii.gz';

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Run vista
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
disp('Vista run in progress ...')
% do_RFs('WIC0326_RF1_vista', subNewTemp, {'ss5', 'func'});
VistaDIR = [subNewTemp '/RF1/WIC0326_RF1_vista'];
do_RFs('WIC0326', VistaDIR , {'ss5'});

% Convert back from volumetric to surface file
VistaOUTPT = [VistaDIR '/RF_ss5-fFit.nii.gz'];
% VistaOUTPT = sprintf('%s/Inplane/Original/RF_%s-fFit.nii.gz',vistaDIR,space);
nii_data = niftiRead(VistaOTPT);
datagii = reshape(nii_data, size(nii_data, 1) * size(nii_data, 2) * size(nii_data, 3), size(nii_data, 4));
fitEstimLeft_gii.cdata = datagii(1:numVerticesLeft, :);
fitEstimRight_gii.cdata = datagii(numVerticesLeft+1:numVerticesLeft+numVerticesRight, :);

% Add the saving thingy for gii here
%LHfFit = sprintf('RF_%s-fFit_hemi-L.func.gii',space);
%gif.cdata = data(1:nvertex_L,:);
%gif = gifti(gif);
%save(gif,LHfFit,'Base64Binary');

% Remove the folder that we just created
rmdir(subNewTemp, 's');

%"${WORKBENCH}" wb_command -set-structure
%"${WORKBENCH}" wb_command -set-map-names


end
