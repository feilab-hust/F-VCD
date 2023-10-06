
function  write2d(data,filename,bitdepth)
[height,width]=size(data);
if bitdepth==16
    data=uint16(data);
end
if bitdepth==8
    data=uint8(data);
end
if bitdepth==32
    data=single(data);
end

t = Tiff(filename,'w');
t.setTag('ImageLength', height);
t.setTag('ImageWidth', width);
t.setTag('Photometric', Tiff.Photometric.MinIsBlack);
t.setTag('BitsPerSample', bitdepth);
t.setTag('SamplesPerPixel', 1);
t.setTag('PlanarConfiguration', Tiff.PlanarConfiguration.Chunky);
if bitdepth==32
 t.setTag('SampleFormat',Tiff.SampleFormat.IEEEFP ); 
end
t.write(data);
t.close;
end
