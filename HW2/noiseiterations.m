function output = noiseiterations(times)
im1 = imread("pout.tif");
[size1, size2] = size(im1); 
tempmat = zeros(size1,size2);
for i = 1:times
    tempnoise = im2double(imnoise(im1,"gaussian",0,0.1));
    tempmat = tempmat + tempnoise;
end
tempmat = tempmat/times;
output = tempmat;
end

