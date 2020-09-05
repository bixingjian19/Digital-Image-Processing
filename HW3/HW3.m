%1  
im1 = imread("autumn.tif");
im2 = imread("football.jpg");
imshowpair(im1,im2,"montage");

%2
h = fspecial("gaussian",[5,5]);

%3
ycbcr = rgb2ycbcr(im1);
Y = ycbcr(:,:,1);

%4 
ycbcr_copy = ycbcr;
Y_conv = imfilter(Y,h,"replicate");
ycbcr_copy(:,:,1) = Y_conv;
subplot(121); imshow(ycbcr2rgb(ycbcr)); title("original")
subplot(122); imshow(ycbcr2rgb(ycbcr_copy)); title("processed");

%5
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

%6
%6.1
% Construct grid of (omega_x, omega_y)
[wx,wy] = meshgrid(-3 : 0.1 : 3, -3 : 0.1 : 3);
% Compute frequency response
type = '2d';
% type = '1d_horizontal';
% type = '1d_vertical');
if strcmp(type, '2d') == 1
    H = (1/25) * (1+2 * cos(wx) + 2 * cos(2 * wx) ).* (1+2 * cos(wy) + 2 * cos(2 * wy) );
    h = 1/25 * ones(5,5);
elseif strcmp(type, '1d_horizontal') == 1
    H = (1+2 * cos(wx) + 2 * cos(2 * wx) ) / 5;
    h = 1/5 * ones(1,5);
elseif strcmp(type, '1d_vertical') == 1
    H = (1+2 * cos(wy) + 2 * cos(2 * wy) ) / 5;
    h = 1/5 * ones(5,1);
end
% Plot magnitude of frequency response
figure(1); clf;
surf(wx/pi, wy/pi, abs(H));
axis([-1 1 -1 1 0 1]);
set(gca, 'XTick', -1 : 1/2 : 1);
set(gca, 'YTick', -1 : 1/2 : 1);
xlabel('\omega_x / \pi'); ylabel('\omega_y / \pi');
zlabel('| H (\omega_x, \omega_y) |');
% Apply to test image
% img = imread('croppedBike.png');
img = imread("pout.tif");
imgFilt = imfilter(img, h, 'symmetric');
figure(2); clf;
subplot(1,2,1); imshow(img); title('Original Image');
subplot(1,2,2); imshow(imgFilt); title('Filtered Image');
%6.2
% Load test image
img = imread('pout.tif');
% Apply two different sharpening filters
h1 = [0, -1, 0; -1, 8, -1; 0, -1, 0] / 4;
h2 = [0, -1, 0; -1, 5, -1; 0, -1, 0];
filteredImg1 = imfilter(img, h1, 'replicate');
filteredImg2 = imfilter(img, h2, 'replicate');
% Show images
figure(1), clf;
subplot(1, 2, 1), imshow(img); title('Original Image');
subplot(1, 2, 2), imshow(filteredImg1); title('Filtered Image');
figure(2), clf;
subplot(1, 2, 1), imshow(img); title('Original Image');
subplot(1, 2, 2), imshow(filteredImg2); title('Filtered Image');
% Define frequency responses
[wx,wy] = meshgrid(-3 : 0.1 : 3, -3 : 0.1 : 3);
H1 = 2 - 0.5 * cos(wx) - 0.5 * cos(wy);
H2 = 5 - 2 * cos(wx) - 2 * cos(wy);
% Show magnitude of frequency response
figure(3), clf;
surf(wx/pi, wy/pi, abs(H1));
set(gca, 'XTick', -1 : 1/2 : 1, 'YTick', -1 : 1/2 : 1);
xlabel('\omega_x / \pi'); ylabel('\omega_y / \pi'); zlabel('| H(\omega_x, \omega_y) |');
figure(4), clf;
surf(wx/pi, wy/pi, abs(H2));
set(gca, 'XTick', -1 : 1/2 : 1, 'YTick', -1 : 1/2 : 1);
xlabel('\omega_x / \pi'); ylabel('\omega_y / \pi'); zlabel('| H(\omega_x, \omega_y) |');
% Save images
imwrite(filteredImg1, 'Sharpening_Filter_1.png');
imwrite(filteredImg2, 'Sharpening_Filter_2.png');

%7
Y1 = ycbcr(:,:,1);

If = fft2(Y1);
keri = zeros(206,345);
keri(1:5,1:5) = h;
kerif = fft2(keri);
Iff = If.*kerif;
Iffi = uint8(ifft2(Iff));

ycbcr_copy1 = ycbcr;
ycbcr_copy1(:,:,1) = Iffi;
subplot(121); imshow(ycbcr2rgb(ycbcr_copy)); title("spatial");
subplot(122); imshow(ycbcr2rgb(ycbcr_copy1)); title("frequency");

%8
image = rgb2gray(im1);
size1 = size(image,1);
size2 = size(image,2);
J = dct2(image);
K1 = idct2(J, round(size1*1,75), round(size2*1.75)); 
K2 = idct2(J, round(size1*0.6), round(size2*0.6)); 
imshowpair(K1,K2,"montage"); title("*1.75(Left) and *0.6(Right)");






