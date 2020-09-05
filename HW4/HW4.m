%1
symbols = (1:6);
prob = [.03 .08 .13 .19 .23 .34]; 
[dict,avglen] = huffmandict(symbols,prob);
dict(:,2) = cellfun(@num2str,dict(:,2),'UniformOutput',false);

%3.
im1 = im2double(imread("ultrasound.jpg"));
im2 = im2double(imread("mammogram.jpg"));
imshowpair(im2uint8(im1),im2uint8(im2),"montage");
title("ultrasound(Left) mammogram(Right)");

%4
ycbcr = rgb2ycbcr(im1);
Y = ycbcr(:,:,1);
imshow(im2uint8(Y));

%5
noise1 = imnoise(Y,"gaussian");
noise2 = imnoise(im2,"gaussian");
imshowpair(im2uint8(noise1),im2uint8(noise2),"montage");

%6
imd = dct2(im2uint8(noise1));
imdl = log10(abs(imd)+1);

s1 = size(Y);
x = [1:s1(1)];
y = [1:s1(2)];
[x,y] = meshgrid(y,x);
figure
surface(x,y,imdl)
view(60,45)

sigma = imd(floor(186*0.75):end,floor(272*0.75):end).*imd(floor(186*0.75),floor(272*0.75):end);
s3 = size(sigma);
xs3 = [1:s3(1)];
ys3 = [1:s3(2)];
[xs3,ys3] = meshgrid(ys3,xs3);
figure
surface(xs3,ys3,sigma)
view(60,45)

NoiseVariance = mean(mean(sigma));
beta = 3.0;
NoiseVariance = beta*NoiseVariance;

%7
SignalVariance = imd.*imd + 0.001;
WienerFilter = 1 + (NoiseVariance./SignalVariance);
WienerFilter = 1./WienerFilter;

FilteredImageDCT = imd.*WienerFilter;
FilteredImage = idct2(FilteredImageDCT);
imo = uint8(FilteredImage);

figure
imshowpair(imo,noise1,'montage');
title('DCT Wiener(left) and Noisy Image (right)')

%8
J = wiener2(noise1);
t = 3;
f = 2;
h1 = 1;
h2 = 20;
selfsim = 0;
denoised = simple_nlm(noise1,t,f,h1,h2,selfsim);
imshowpair(J,denoised,"montage"); title("wiener(Left) & nonelocal means(Right)");

%9
face = imread("pout.tif");
face = imresize(face, [128 128]);
[c,s]=wavedec2(face,6,'haar');

[H1,V1,D1] = detcoef2('all',c,s,1);
A1 = appcoef2(c,s,'haar',1);

V1img = wcodemat(V1,255,'mat',1);
H1img = wcodemat(H1,255,'mat',1);
D1img = wcodemat(D1,255,'mat',1);
A1img = wcodemat(A1,255,'mat',1);

subplot(2,2,1)
imagesc(A1img)
colormap pink(255)
title('Approximation Coef. of Level 1')

subplot(2,2,2)
imagesc(H1img)
title('Horizontal Detail Coef. of Level 1')

subplot(2,2,3)
imagesc(V1img)
title('Vertical Detail Coef. of Level 1')

subplot(2,2,4)
imagesc(D1img)
title('Diagonal Detail Coef. of Level 1')


%10
[a2,h2,v2,d2] = haart2(face,2);
imagesc(a2)

X =  imread('lena.png'); 
X = X(:,:,2);

[c,s]=wavedec2(X,4,'haar');
[H1,V1,D1] = detcoef2('all',c,s,1);
A1 = appcoef2(c,s,'haar',1); 
V1img = wcodemat(V1,255,'mat',1);
H1img = wcodemat(H1,255,'mat',1);
D1img = wcodemat(D1,255,'mat',1);
A1img = wcodemat(A1,255,'mat',1);

[H2,V2,D2] = detcoef2('all',c,s,2);
A2 = appcoef2(c,s,'haar',2); 
V2img = wcodemat(V2,255,'mat',1);
H2img = wcodemat(H2,255,'mat',1);
D2img = wcodemat(D2,255,'mat',1);
A2img = wcodemat(A2,255,'mat',1);

[H3,V3,D3] = detcoef2('all',c,s,3);
A3 = appcoef2(c,s,'haar',3); 
V3img = wcodemat(V2,255,'mat',1);
H3img = wcodemat(H2,255,'mat',1);
D3img = wcodemat(D2,255,'mat',1);
A3img = wcodemat(A2,255,'mat',1);

[H2,V2,D2] = detcoef2('all',c,s,4);
A4 = appcoef2(c,s,'haar',4); 
V4img = wcodemat(V2,255,'mat',1);
H4img = wcodemat(H2,255,'mat',1);
D4img = wcodemat(D2,255,'mat',1);
A4img = wcodemat(A2,255,'mat',1);


subplot(2,2,1);
imagesc(A1img);
colormap pink(255);
title('Approximation Coef. of Level 1');

subplot(2,2,2);
imagesc(H1img);
title('Horizontal detail Coef. of Level 1');

subplot(2,2,3);
imagesc(V1img);
title('Vertical detail Coef. of Level 1');

subplot(2,2,4);
imagesc(D1img);
title('Diagonal detail Coef. of Level 1');


%***********************************

figure;
subplot(2,2,1);
imagesc(A2img);
colormap pink(255);
title('Approximation Coef. of Level 2');

subplot(2,2,2)
imagesc(H2img);
title('Horizontal detail Coef. of Level 2');

subplot(2,2,3)
imagesc(V2img);
title('Vertical detail Coef. of Level 2');

subplot(2,2,4)
imagesc(D2img);
title('Diagonal detail Coef. of Level 2');
%***********************************

figure;
subplot(2,2,1);
imagesc(A3img);
colormap pink(255);
title('Approximation Coef. of Level 3');

subplot(2,2,2)
imagesc(H3img);
title('Horizontal detail Coef. of Level 3');

subplot(2,2,3)
imagesc(V3img);
title('Vertical detail Coef. of Level 3');

subplot(2,2,4)
imagesc(D3img);
title('Diagonal detail Coef. of Level 3');
%***********************************

figure;
subplot(2,2,1);
imagesc(A4img);
colormap pink(255);
title('Approximation Coef. of Level 4');

subplot(2,2,2)
imagesc(H4img);
title('Horizontal detail Coef. of Level 4');

subplot(2,2,3)
imagesc(V4img);
title('Vertical detail Coef. of Level 4');

subplot(2,2,4)
imagesc(D4img);
title('Diagonal detail Coef. of Level 4');






