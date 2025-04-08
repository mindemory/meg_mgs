      function [x, cmat, jacobian, rms, transpoints] = calc_transform_lm_seq(coord1pts, coord2pts, x_init)
%Runs L-M optimization on arm data and manually determined points from
%sequoia
%images to determine the optimal transformation matrix (cmat) from the
%ultrasound image plane coordinate system to the arm tip coordinate system.
%Inputs:
%arm_out:  data structure containing data from calibration arm files
%coord2pts: physical coordinates in mm (specified relative to origin
%point specified during point selection) of the cross-wire intersection determined from sequoia files in
%'get_points_seq' script
%x_init: initial estimates for 9 calbration parameters, stored in a row vector
%x_init(1)-x_init(3) are approximations of translational offsets in cmat (transformation matrix from
%image plane coordinate system to arm tip coordinate system)
%x(1)- x-translational offset from armtip to origin of image plane coord
%system
%x(2)- y-translational offset from armtip to origin of image plane coord
%system
%x(3)- z-translational offset from armtip to origin of image plane coord
%system
%x_init(4)-x_init(6) are approximations of rotational parameters specifying transformation from image
%plane coord system to arm tip coord system
%x(6)- rotation about x-axis of image plane coordinate system (done first)
%x(5)- rotation about y-axis of image plane coordinate system after first rotation (done second)
%x(4)- rotation about z-axis of image plane coordinate system after first and second rotation (done third)
%x_init(7)-xinit(9)-  approximations of translational offsets in rmat (matrix transformation from
%global arm coordinate system to global coordinate system with its origin
%at cross-wire intersection.)  These 3 parameters can be estimated at the
%time of calibration by attempting to register the location of the
%cross-wire intersection with the arm.
%x(7)- x-translational offset from global arm coordinate system to
%cross-wire intersection
%x(8)- y-translational offset from global arm coordinate system to
%cross-wire intersection
%x(9)- z-translational offset from global arm coordinate system to
%cross-wire intersection
%Outputs:
%x- optimized solution for vector x 
%rmat- optimized solution for rmat matrix
%cmat- optimized solution for cmat matrix, what we are mainly interested in
%rms- vector containing distance in mm of each reconstructed manualpoint from
%(0,0,0) coordinate that it should ideally reconstruct to.
%jacobian-measure of how well-conditioned the calibration equations are is
%condition number of jacobian
%S. Dandekar



          
                                         
                                     
       
    x=x_init; %initially set x equal to initial estimates
                %for calibration parameters provided by user
                                                      
                                                      
       %define initial estimate for transformation matrix
       %from image plane to arm tip
       cmatinit.str(1,1).str= '(cos(x(4))*cos(x(5)))';
       cmatinit.str(1,2).str= '(cos(x(4))*sin(x(5))*sin(x(6))-sin(x(4))*cos(x(6)))';
       cmatinit.str(1,3).str= '(cos(x(4))*sin(x(5))*cos(x(6))+ sin(x(4))*sin(x(6)))';
       cmatinit.str(1,4).str= '(x(1))';

       cmatinit.str(2,1).str= '(sin(x(4))*cos(x(5)))';
       cmatinit.str(2,2).str= '(sin(x(4))*sin(x(5))*sin(x(6))+cos(x(4))*cos(x(6)))';
       cmatinit.str(2,3).str= '(sin(x(4))*sin(x(5))*cos(x(6))-cos(x(4))*sin(x(6)))';
       cmatinit.str(2,4).str= '(x(2))';
       
       cmatinit.str(3,1).str= '(-sin(x(5)))';
       cmatinit.str(3,2).str= '(cos(x(5))*sin(x(6)))';
       cmatinit.str(3,3).str= '(cos(x(5))*cos(x(6)))';
       cmatinit.str(3,4).str= '(x(3))';
       
       
       cmatinit.str(4,1).str= '0';
       cmatinit.str(4,2).str= '0';                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                
       cmatinit.str(4,3).str= '0';
       cmatinit.str(4,4).str= '1';
       
       
%    
%        %initial estimates for reconstruction transformation matrix
%        %(with crosswire phantom, set 3 orientation values to 0 (Prager p.859)
%         rmatinit.str(1,1).str= '1';
%         rmatinit.str(1,2).str= '0';
%         rmatinit.str(1,3).str= '0';
%         rmatinit.str(1,4).str= 'x(7)';
% 
%         rmatinit.str(2,1).str= '0';
%         rmatinit.str(2,2).str= '1';
%         rmatinit.str(2,3).str= '0';
%         rmatinit.str(2,4).str= 'x(8)';
%        
%         rmatinit.str(3,1).str= '0';
%         rmatinit.str(3,2).str= '0';
%         rmatinit.str(3,3).str= '1';
%         rmatinit.str(3,4).str= 'x(9)';
%         
%        
%         rmatinit.str(4,1).str= '0';
%         rmatinit.str(4,2).str= '0';
%         rmatinit.str(4,3).str= '0';
%         rmatinit.str(4,4).str= '1';
        
   
 
          
%form string input of equations to input to optimization function
   
f= '['; 
     
    %get the transformation matrix from arm tip to global arm coordinates for each
    %of the calibration images:
    for pointind= 1:size(coord2pts,1) 
    
%        for i=1:4
%            for j=1:4
%          tmat.str(i,j).str=num2str(arm_out.ArmFrame(frameind).tmatrix(i,j));
%        end
%      end
        
     %form the column vector of values [sxu syv 0 1]T
         mpoints.str(1,1).str=num2str(coord2pts(pointind, 1));
         mpoints.str(2,1).str=num2str(coord2pts(pointind, 2));
         mpoints.str(3,1).str=num2str(coord2pts(pointind, 3));
         mpoints.str(4,1).str='1';
       
         
        rmat.str(1,1).str= '1';
        rmat.str(1,2).str= '0';
        rmat.str(1,3).str= '0';
        rmat.str(1,4).str= num2str(-coord1pts(pointind, 1));

        rmat.str(2,1).str= '0';
        rmat.str(2,2).str= '1';
        rmat.str(2,3).str= '0';
        rmat.str(2,4).str= num2str(-coord1pts(pointind, 2));
       
        rmat.str(3,1).str= '0';
        rmat.str(3,2).str= '0';
        rmat.str(3,3).str= '1';
        rmat.str(3,4).str=  num2str(-coord1pts(pointind, 3));
        
       
        rmat.str(4,1).str= '0';
        rmat.str(4,2).str= '0';
        rmat.str(4,3).str= '0';
        rmat.str(4,4).str= '1';
        
         
         
         
         
         
         
         
         
     %Do multiplication of the 3 4x4s and one 4x1 
       prod1 = str_mult_4x1(cmatinit, mpoints);
       prod1.str(1,1).str
    
%  **      prod2 = str_mult_4x1(tmat, prod1);
%   **     prod = str_mult_4x1(rmatinit, prod2) ;

         prod = str_mult_4x1(rmat, prod1) ;
       
       %get the top 3 rows of the result of the multiplication for the optimization
       %(3 calibration equations per manualpoint)
       f= strcat(f, prod.str(1,1).str, ',', prod.str(2,1).str, ',', prod.str(3,1).str);
     
       if (pointind ~= (size(coord2pts, 1))) 
           f= strcat(f, ','); 
           f 
         %  pause
       
        end
    end
f=strcat(f, ']')  ;     
     
      
     %optimization options:
      OPTIONS=optimset('LargeScale','off', 'LevenbergMarquardt','on', 'Jacobian','off', 'MaxFunEvals', 10000000);
      
      %apply optimization 
      [x,resnorm,residual,exitflag,output,lambda,jacobian]= lsqnonlin({f},x,[],[] ,OPTIONS);
      fval = resnorm;
    
     %get RMS errors (i.e. deviations in solution from reconstructing to
     %ideal (0,0,0) point
      for i=1:3:length(residual)
          
          rms((i-1)/3 +1)= sqrt((residual(i))^2 + (residual(i+1))^2 + (residual(i+2))^2);
      
      end
      

      
      
      rms
      mean(rms)
      std(rms)
      median(rms)
      max(rms)
      
      
       disp(sprintf('\nValue of the function at the solution: %g', fval) ); 
      
       
  %calculate actual values of calibration matrix based on LM solution:
  
       cmat(1,1)= (cos(x(4))*cos(x(5)));
       cmat(1,2)= (cos(x(4))*sin(x(5))*sin(x(6))-sin(x(4))*cos(x(6)));
       cmat(1,3)= (cos(x(4))*sin(x(5))*cos(x(6))+ sin(x(4))*sin(x(6)));
       cmat(1,4)= x(1);

       cmat(2,1)= (sin(x(4))*cos(x(5)));
       cmat(2,2)= (sin(x(4))*sin(x(5))*sin(x(6))+cos(x(4))*cos(x(6)));
       cmat(2,3)= (sin(x(4))*sin(x(5))*cos(x(6))- cos(x(4))*sin(x(6)));
       cmat(2,4)= x(2);
       
       cmat(3,1)= (-sin(x(5)));
       cmat(3,2)= (cos(x(5))*sin(x(6)));
       cmat(3,3)= (cos(x(5))*cos(x(6)));
       cmat(3,4)= x(3);
       
       
       cmat(4,1)= 0;
       cmat(4,2)= 0;
       cmat(4,3)= 0;
       cmat(4,4)= 1;
  
       
       
       
       
       for pointind= 1:size(coord2pts,1) 
       
       transpoints(pointind, :)= cmat*[coord2pts(pointind, :) 1]';
           
           
           
       end
       transpoints=transpoints(:, 1:3);
       
       
  
       %final estimates 
%        %(with crosswire phantom, set 3 orientation values to 0 (Prager p.859)
%         rmat(1,1)= 1;
%         rmat(1,2)= 0;
%         rmat(1,3)= 0;
%         rmat(1,4)= x(7);
% 
%         rmat(2,1)= 0;
%         rmat(2,2)= 1;
%         rmat(2,3)= 0;
%         rmat(2,4)= x(8);
%        
%         rmat(3,1)= 0;
%         rmat(3,2)= 0;
%         rmat(3,3)= 1;
%         rmat(3,4)= x(9);
%         
%        
%         rmat(4,1)= 0;
%         rmat(4,2)= 0;
%         rmat(4,3)= 0;
%         rmat(4,4)= 1;
%          
%         x
%         rmat
%         cmat
% 
%         
%   
%  
  
  
  
  
  

   

