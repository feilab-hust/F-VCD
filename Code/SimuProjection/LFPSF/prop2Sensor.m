% oLaF - a flexible 3D reconstruction framework for light field microscopy
% Copyright (c)2017-2020 Anca Stefanoiu and Josue Page

function f1 = prop2Sensor(f0, sensorRes, z, lambda, idealSampling)

%% Computes the final Lightfield PSF
if (idealSampling)
    if (z == 0)
        f1 = f0;
    else
        %% Ideal sampling rate (compute the impulse response h) -> computational Fourier Optics book
        Lx = size(f0,1)*sensorRes(1);
        Ly = size(f0,2)*sensorRes(2);
        k = 2*pi/lambda;
        
        % ideal_rate = [lambda*z/Lx, lambda*z/Ly]; % Fresnel
        ideal_rate = [lambda*sqrt(z^2 +(Lx/2)^2)/Lx, lambda*sqrt(z^2 +(Ly/2)^2)/Ly];
        ideal_samples_no = ceil([Lx Ly]./ideal_rate);
        ideal_samples_no = ideal_samples_no + (1-mod(ideal_samples_no,2)); %odd
        rate = [Lx, Ly]./ideal_samples_no; %lambda*sqrt(z^2 +(Lx/2)^2)/Lx
        
        % spacial frequencies in x and y direction
        du = 1./(ideal_samples_no(1)*single(rate(1))); %1/Lx
        dv = 1./(ideal_samples_no(2)*single(rate(2))); 
        u = [0:ceil(ideal_samples_no(1)/2)-1 ceil(-ideal_samples_no(1)/2):-1]*du; 
        v = [0:ceil(ideal_samples_no(2)/2)-1 ceil(-ideal_samples_no(2)/2):-1]*dv; 

        % transfer function for Rayleigh-Sommerfeld diffraction integral
        H = exp(1i*sqrt(1-lambda^2*(repmat(u',1,length(v)).^2+repmat(v,length(u),1).^2))*z*k); 

        f1 = ((ifft2( fft2((imresize(f0, ideal_samples_no, 'bicubic'))) .* H ))); 
        f1 = imresize(f1, size(f0), 'bicubic');
    end
else
   if (z == 0)
        f1 = f0;
   else
        %% Original Sampling: compute the Transfer Function H
        Nx = size(f0,1);
        Ny = size(f0,2);
        k = 2*pi/lambda;

        % spacial frequencies in x and y direction
        du = 1./(Nx*single(sensorRes(1)));
        dv = 1./(Ny*single(sensorRes(2)));
        u = [0:ceil(Nx/2)-1 ceil(-Nx/2):-1]*du; 
        v = [0:ceil(Ny/2)-1 ceil(-Ny/2):-1]*dv; 

        % transfer function for Rayleigh diffraction integral
        H = exp(1i*sqrt(1-lambda^2*(repmat(u',1,length(v)).^2+repmat(v,length(u),1).^2))*z*k); 

        % final Lightfield PSF -> sensor image
        f1 = exp(1i*k*z)*(ifft2(fft2((f0)) .* H ));
   end
end 