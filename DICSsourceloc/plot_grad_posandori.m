function plot_grad_posandori(grad)

for i=1:size(grad.pnt, 1)
    scatter3(grad.pnt(i, 1), grad.pnt(i, 2), grad.pnt(i, 3));
    
    hold on;
    x(1, :)=grad.pnt(i,1);
    x(2, :)=grad.pnt(i, 1)+grad.ori(i, 1);
    y(1, :)=grad.pnt(i,2);
    y(2, :)=grad.pnt(i, 2)+grad.ori(i, 2)
    z(1, :)=grad.pnt(i,3);
    z(2, :)=grad.pnt(i,3)+grad.ori(i, 3);
    
  
    line(x,y,z);
   
end 


end