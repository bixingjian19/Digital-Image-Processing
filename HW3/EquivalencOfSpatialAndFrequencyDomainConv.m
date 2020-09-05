close all
clear all
clc

% I = imread('cameraman.tif');
I = imresize(rgb2gray(imread("autumn.tif")),[256,256]);
imshow(I)
title('original image')

ker = -ones(3,3);
ker(2,2) = 9;
Ik = uint8(conv2(I,ker,'same'));
Ikf = uint8(conv2(I,ker));

figure
imshow(Ikf)
title('convolved image full')


figure
imshow(Ik)

title('convolved image same size as input')

If = fft2(I);
keri = zeros(256,256);
keri(1:3,1:3) = ker;
kerif = fft2(keri);
Iff = If.*kerif;
Iffi = uint8(ifft2(Iff));

figure
imshow(Iffi)

title('FFT convolved image')


Idiff = uint8(Ikf(1:256,1:256) - Iffi);

max(max(Idiff))
min(min(Idiff))


figure
imshow(Idiff)