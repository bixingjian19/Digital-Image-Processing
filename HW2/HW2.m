%1
im1 = imread("autumn.tif");
im2 = imread("football.jpg");
im3 = imread("pout.tif");
im4 = imread("cameraman.tif");
subplot(141); imshow(im1);
subplot(142); imshow(im2);
subplot(143); imshow(im3);
subplot(144); imshow(im4);


%2
size(im1)
size(im2)
size(im3)
size(im4)


%3
LAB = rgb2lab(im1);
XYZ = rgb2xyz(im1);
YCBCR = rgb2ycbcr(im1);
LAB1 = lab2rgb(LAB);
XYZ1 = xyz2rgb(XYZ);
YCBCR1 = ycbcr2rgb(YCBCR);
subplot(131); imshow(LAB); title("LAB");
subplot(132); imshow(XYZ); title("XYZ");
subplot(133); imshow(YCBCR); title("YCBCR");
subplot(131); imshow(LAB1); 
subplot(132); imshow(XYZ1);
subplot(133); imshow(YCBCR1);


%4
YCBCR_copy = YCBCR;
Y = YCBCR(:,:,1);
H = fspecial('average');
smoothing = imfilter(Y,H,'replicate');
YCBCR_copy(:,:,1) = smoothing;
subplot(121); imshow(ycbcr2rgb(YCBCR)); title("original")
subplot(122); imshow(ycbcr2rgb(YCBCR_copy)); title("smoothing");

YCBCR_copy = YCBCR;
Y = YCBCR(:,:,1);
H = fspecial('sobel');
sharpening = imfilter(Y,H,'replicate');
YCBCR_copy(:,:,1) = sharpening;
subplot(121); imshow(ycbcr2rgb(YCBCR)); title("original")
subplot(122); imshow(ycbcr2rgb(YCBCR_copy)); title("sharpening");


%5
gaussian = imnoise(im3,"gaussian");
saltpepper = imnoise(im3,"salt & pepper");
subplot(121); imshow(gaussian); title("gaussian noise");
subplot(122); imshow(saltpepper); title("salt & pepper noise");


%6
h = fspecial("average");
gaussian_remove = imfilter(gaussian,h,"replicate");
imshowpair(gaussian,gaussian_remove,"montage");
sp_remove = medfilt2(saltpepper);
imshowpair(saltpepper,sp_remove,"montage");


%7
colormap(gray)
image = 100*ones(100);
image(50:100,:) = 50;
image(:,50:100) = 2*image(:,50:100);
fs = fspecial('average');
image = imfilter(image,fs,'symmetric');

sigma = 10;
noisy = image + sigma*randn(size(image));

t = 3;
f = 2;
h1 = 1;
h2 = 20;
selfsim = 0;

denoised = simple_nlm(noisy,t,f,h1,h2,selfsim);

figure(1)
subplot(2,2,1),imagesc(image),title('original');
subplot(2,2,2),imagesc(noisy),title('noisy');
subplot(2,2,3),imagesc(denoised),title('filtered');
gaussian_remove_compare = filter2(fspecial('average'),noisy)/255;
subplot(2,2,4),imagesc(gaussian_remove_compare),title('averaging filter compare');


%8
red = imread("red.jpg");
blue = imread("blue.jpg");

R = red(:,:,1); 
G = red(:,:,2);      
B = red(:,:,3);

Rave = mean(mean(R)); 
Gave = mean(mean(G)); 
Bave = mean(mean(B));
Ave = (Rave + Gave + Bave) / 3;

R1 = (Ave/Rave)*R; G1 = (Ave/Gave)*G; B1 = (Ave/Bave)*B; 
RGB_white = cat(3, R1, G1, B1);
RGB_balanced = uint8(RGB_white); 
imshowpair(red,RGB_balanced,"montage");

output = scalebymax(blue);
imshowpair(blue,output,"montage");






