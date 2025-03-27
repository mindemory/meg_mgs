function surf        = readPial(pialPath)

% Convert pial to mat before running this script
% from terminal run: 
%           mris_convert lh.pial lh.pial.asc
%           mris_convert rh.pial rh.pial.asc

fid                  = fopen(pialPath, 'r');
fgetl(fid); % Remove line 1 which will be "#!ascii verion of xh.pial"
initData             = fscanf(fid, '%f', [inf]);
numVertices          = initData(1);
numFaces             = initData(2);
fclose(fid);
fid                  = fopen(pialPath, 'r');
fgetl(fid); fgetl(fid); % Remove the second line and reald again
pialData             = fscanf(fid, '%f', [4, inf])';
fclose(fid);
vertices             = pialData(1:numVertices, 1:3);
faces                = pialData(numVertices+1:end, 1:3) + 1;
surf.vertices        = vertices;
surf.faces           = faces;

end