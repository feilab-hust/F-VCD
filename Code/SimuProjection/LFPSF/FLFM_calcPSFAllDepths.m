% oLaF - a flexible 3D reconstruction framework for light field microscopy
% Copyright (c)2017-2020 Anca Stefanoiu & Josuue Page

function psfSTACK = FLFM_calcPSFAllDepths(Camera, Resolution)
%   Computes PSF for all Resolution.depths, exploiting the symetry of Resolution.depths at
%   the same absolute distance to the zero plane

psfSTACK = zeros(length(Resolution.yspace), length(Resolution.xspace), length(Resolution.depths));
for i = 1:length(Resolution.depths)
    computePsf = 1;
    
    % Check if the abs(depth) was previoulsy computed, as zero-symetric depths are just conjugates.
    idx = find(abs(Resolution.depths(1:i-1)) == abs(Resolution.depths(i)));
    if ~isempty(idx)
        computePsf = 0;
    end
    
    % If depth has not been computed, compute it
    if computePsf == 1
        % the psf is translationally invariant, so it is sufficient to compute in the origin
        psfWAVE = FLFM_calcPSF(0, 0, Resolution.depths(i), Camera, Resolution);
    else
        % if it is exactly the same depth just copy it
        if Resolution.depths(i) == Resolution.depths(idx)
            psfWAVE = psfSTACK(:,:,idx);
        else
            % if it is the negative, conjugate
            psfWAVE = conj(psfSTACK(:,:,idx));
        end
    end
    psfSTACK(:,:,i)  = psfWAVE; 
end
end
