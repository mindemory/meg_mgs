function [fidpol, sphereptsout, origin, grad, proj, template]=get_markers_and_spherepts_MEGcoords(sfpfile, posfile, radius)
%sfpfile and posfile- as gotten from MEG160 BESA export
%radius- radius in cm as determined by sphere fit in MEG 160
t=loadtxt(sfpfile)

markerrows=[1:5]
sphererows=6:size(t,1);

for m=1:length(markerrows)
fidpol(m,1)=t{m, 2}./10;
fidpol(m, 2)=t{m, 3}./10;
fidpol(m, 3)=t{m, 4}./10;
end

for ns=1:length(sphererows)
spherepts(ns, 1)=t{sphererows(ns), 2}./10;
spherepts(ns, 2)=t{sphererows(ns), 3}./10;
spherepts(ns, 3)=t{sphererows(ns), 4}./10;
end

p=loadtxt(posfile);
ngrads=size(p, 1);

for g=1:ngrads
   grad.pnt(g, 1)=p{g,1}./10;
   grad.pnt(g, 2)=p{g,2}./10;
   grad.pnt(g, 3)=p{g, 3}./10;
  
   grad.ori(g, 1)=p{g,7};
   grad.ori(g, 2)=p{g,8};
   grad.ori(g, 3)=p{g,9};
   
end

for g=(ngrads+1):(ngrads*2) %define second coils in pair
   grad.pnt(g, 1)=p{g-ngrads,4}./10;
   grad.pnt(g, 2)=p{g-ngrads,5}./10;
   grad.pnt(g, 3)=p{g-ngrads,6}./10;   

   grad.ori(g, 1)=-p{g-ngrads,7};
   grad.ori(g, 2)=-p{g-ngrads,8};
   grad.ori(g, 3)=-p{g-ngrads,9};
   
   gvectx(g-ngrads)=grad.pnt(g, 1)-grad.pnt(g-ngrads, 1);
   gvecty(g-ngrads)=grad.pnt(g,2)-grad.pnt(g-ngrads, 2);
   gvectz(g-ngrads)=grad.pnt(g,3)-grad.pnt(g-ngrads,3);
   
   
   
end


% Define the pair of 1st and 2nd coils for each gradiometer
grad.tra = repmat(diag(ones(1,size(grad.pnt,1)/2),0),1,2);

% Make the matrix sparse to speed up the multiplication in the forward
% computation with the coil-leadfield matrix to get the channel leadfield
grad.tra = sparse(grad.tra);


grad.label = cell(ngrads, 1);% cell-array containing strings, Nchan X 1 (This is one modification to yokogawa2grad.m -Sangi)
for i=1:ngrads
  grad.label{i} = num2str(i);
end
grad.unit='cm';

origin=fminunc(@(origin) sum(((spherepts(:,1)-origin(1)).^2 +(spherepts(:,2)-origin(2)).^2+(spherepts(:,3)-origin(3)).^2-radius^2).^2), [0; 0; 0])

[theta, phi]=meshgrid(0:.1:2*pi, 0:.1:pi);

x=radius.*cos(theta).*sin(phi)+origin(1);
y=radius*sin(theta).*sin(phi)+origin(2);
z=radius*cos(phi)+origin(3);

ncols=size(theta, 2);

sphereptsoutx=[];
sphereptsouty=[];
sphereptsoutz=[];

for col=1:ncols
    sphereptsoutx=[sphereptsoutx; x(:,col)];
    sphereptsouty=[sphereptsouty; y(:,col)];
    sphereptsoutz=[sphereptsoutz; z(:,col)];
end

sphereptsout(:,1)=sphereptsoutx;
sphereptsout(:,2)=sphereptsouty;
sphereptsout(:,3)=sphereptsoutz;


proj=zeros(ngrads,3);

for g=(ngrads+1):(ngrads*2)
 multvect=fminunc(@(multvect) sum((((grad.pnt(g, 1)+multvect*gvectx(g-ngrads))-origin(1)).^2 +((grad.pnt(g, 2)+multvect*gvecty(g-ngrads))-origin(2)).^2+((grad.pnt(g, 3)+multvect*gvectz(g-ngrads))-origin(3)).^2-radius^2).^2), [1])
 proj(g-ngrads, 1)= (grad.pnt(g, 1)+multvect*gvectx(g-ngrads));  
 proj(g-ngrads, 2)=(grad.pnt(g, 2)+multvect*gvecty(g-ngrads));
 proj(g-ngrads, 3)=(grad.pnt(g,3)+multvect*gvectz(g-ngrads));
 
end


 template.grad.pnt(1:157, 1)=proj(:,1);
 template.grad.pnt(1:157, 2)=proj(:,2);
 template.grad.pnt(1:157, 3)=proj(:,3);
 
 template.grad.pnt(158:314, 1)=proj(:,1)+gvectx(1:157)';
 template.grad.pnt(158:314, 2)=proj(:,2)+gvecty(1:157)';
 template.grad.pnt(158:314, 3)=proj(:,3)+gvectz(1:157)';

template.grad.ori=grad.ori;
template.grad.tra=grad.tra;
template.grad.label=grad.label;
template.grad.unit=grad.unit;










