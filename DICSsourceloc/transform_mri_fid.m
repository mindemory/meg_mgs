function transpoints=transform_mri_fid(fidmri, mri)  
       

cmat=mri.transform;

cmat(1:3, 4)=cmat(1:3,4)./10;


for pointind= 1:size(fidmri,1) 
       
 transpoints(pointind, :)= cmat*[fidmri(pointind, :) 1]';
           
           
           
end
transpoints=transpoints(:, 1:3);