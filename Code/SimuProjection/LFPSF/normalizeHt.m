% oLaF - a flexible 3D reconstruction framework for light field microscopy
% Copyright (c)2017-2020 Anca Stefanoiu and Josue Page

function Ht = normalizeHt(Ht)
for aa = 1:size(Ht,1)
    for bb = 1:size(Ht,2)
        temp = cat(2,Ht{aa,bb,:});
        sum_temp = sum(temp(:));
        if(sum_temp==0)
            continue
        end
        % Normalize such that all the backward patterns (for all depths) of
        % a single point in the sensor sum to 1
        for c = 1:size(Ht,3)
            Ht{aa,bb,c} = sparse(Ht{aa,bb,c}/sum_temp);
        end
    end
end