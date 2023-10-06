% oLaF - a flexible 3D reconstruction framework for light field microscopy
% Copyright (c)2017-2020 Anca Stefanoiu and Josue Page

function H = ignoreSmallVals(H, tol)    
for c = 1: size(H,3)
    for aa = 1:size(H,1)
        for bb = 1:size(H,2)
            temp = H{aa,bb,c};
            max_slice = max(temp(:));
            % Clamp values smaller than tol 
            temp(temp < max_slice*tol) = 0;
            sum_temp = sum(temp(:));
            if(sum_temp==0)
                continue
            end
            % and normalize per individual PSF such that the sum is 1.
            temp = temp/sum_temp;
            H{aa,bb,c} = sparse(temp);
        end
    end
end