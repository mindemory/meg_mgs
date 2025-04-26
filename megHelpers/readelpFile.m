function [fidData, hpiData]     = readelpFile(fPath)
fidStarters                     = ["%F"];
hpiStarters                     = ["RED", "YELLOW", "BLUE", "WHITE", "BLACK"];
dataRead                        = readlines(fPath);

fidLineIdx                      = [];
hpiLineIdx                      = [];
for i                           = 1:length(dataRead)
    if contains(dataRead(i), fidStarters)
        fidLineIdx              = [fidLineIdx, i];
    elseif contains(dataRead(i), hpiStarters)
        hpiLineIdx              = [hpiLineIdx, i+1];
    end
end

% Extract fiducial data
fidData                         = NaN(length(fidLineIdx), 3);
for lIdx                        = 1:length(fidLineIdx)
    thLineData                  = split(dataRead(fidLineIdx(lIdx)));
    fidData(lIdx, :)            = str2double(thLineData(2:4));

end

% Extract HPI data
hpiData                        = NaN(length(hpiLineIdx), 3);
for hIdx                       = 1:length(hpiLineIdx)
    thLineData                 = split(dataRead(hpiLineIdx(hIdx)));
    hpiData(hIdx, :)           = str2double(thLineData);
end



end