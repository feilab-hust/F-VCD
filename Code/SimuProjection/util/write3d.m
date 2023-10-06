function write3d(data,filename,bitdepth)
[height,width,depth]=size(data);

if bitdepth==16
    data=uint16(data);
end
if bitdepth==8
    data=uint8(data);
end
if bitdepth==32
    data=single(data);
end
for j = 1:1:depth 
    if j == 1
        t = Tiff(filename,'w');
    else
        t = Tiff(filename, 'a');
    end

    t.setTag('ImageLength', height);
    t.setTag('ImageWidth', width);
    t.setTag('Photometric', Tiff.Photometric.MinIsBlack);
    t.setTag('BitsPerSample', bitdepth);
    t.setTag('SamplesPerPixel', 1);
    t.setTag('PlanarConfiguration', Tiff.PlanarConfiguration.Chunky);
    
     if bitdepth==32
         t.setTag('SampleFormat',Tiff.SampleFormat.IEEEFP ); 
    end
    t.write(data(:,:,j));
    t.close;
end
end