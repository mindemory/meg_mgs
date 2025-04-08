function fidpol=get_pol_fid(elpfile)

markerrows=[17 22 27 32 37]
for m=1:length(markerrows)
t=loadtxt(elpfile, 'skipline', markerrows(m)-1, 'nlines', 1);
fidpol(m,1)=t{1};
fidpol(m, 2)=t{2};
fidpol(m, 3)=t{3};
end
markerrows=[17 22 27 32 37]
for m=1:length(markerrows)
t=loadtxt(elpfile, 'skipline', markerrows(m)-1, 'nlines', 1);
fidpol(m,1)=t{1}.*100;
fidpol(m, 2)=t{2}.*100;
fidpol(m, 3)=t{3}.*100;
end