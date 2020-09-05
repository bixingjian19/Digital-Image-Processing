%1
im1 = im2double(imread("mammogram.jpg"));
im2 = im2double(imread("ultrasound.jpg"));
im3 = im2double(imread("sky2.jpg"));
im4 = im2double(imread("skyscraper2.jpg"));
subplot(221); imshow(im2uint8(im1)); title("mammogram");
subplot(222); imshow(im2uint8(im2)); title("ultrasound");
subplot(223); imshow(im2uint8(im3)); title("building 1");
subplot(224); imshow(im2uint8(im4)); title("building 2");

%2
im1_ycbcr = rgb2ycbcr(im1);im1_ycomp = im1_ycbcr(:,:,1);
im2_ycbcr = rgb2ycbcr(im2);im2_ycomp = im2_ycbcr(:,:,1);
im3_ycbcr = rgb2ycbcr(im3);im3_ycomp = im3_ycbcr(:,:,1);
im4_ycbcr = rgb2ycbcr(im4);im4_ycomp = im4_ycbcr(:,:,1);
subplot(221); imshow(im2uint8(im1_ycomp)); title("mammogram(Y component)");
subplot(222); imshow(im2uint8(im2_ycomp)); title("ultrasound(Y component)");
subplot(223); imshow(im2uint8(im3_ycomp)); title("building 1(Y component)");
subplot(224); imshow(im2uint8(im4_ycomp)); title("building 2(Y component)");

%3
im1_noise = imnoise(im1_ycomp,"gaussian",0,0.05);
im2_noise = imnoise(im2_ycomp,"gaussian",0,0.05);
im3_noise = imnoise(im3_ycomp,"gaussian",0,0.1);
im4_noise = imnoise(im4_ycomp,"gaussian",0,0.1);
subplot(221); imshow(im2uint8(im1_noise)); title("mammogram with Gaussian noise");
subplot(222); imshow(im2uint8(im2_noise)); title("ultrasound with Gaussian noise)");
subplot(223); imshow(im2uint8(im3_noise)); title("building 1 with Gaussian noise");
subplot(224); imshow(im2uint8(im4_noise)); title("building 2 with Gaussian noise");

%4
dct_im1 = dct2(im1_noise);
s = size(im1_noise);
sigma1 = dct_im1(floor((s(1)*0.75)):end,floor((s(2)*0.75)):end).*dct_im1(floor((s(1)*0.75)):end,floor((s(2)*0.75)):end);
sigmasize = size(sigma1);
xs = [1:sigmasize(1)];ys = [1:sigmasize(2)];
[xs,ys] = meshgrid(ys,xs);
figure; surface(xs,ys,sigma1); view(60,45); title("noise sample 1");

dct_im2 = dct2(im2_noise);
s = size(im2_noise);
sigma2 = dct_im2(floor((s(1)*0.75)):end,floor((s(2)*0.75)):end).*dct_im2(floor((s(1)*0.75)):end,floor((s(2)*0.75)):end);
sigmasize = size(sigma2);
xs = [1:sigmasize(1)];ys = [1:sigmasize(2)];
[xs,ys] = meshgrid(ys,xs);
figure; surface(xs,ys,sigma2); view(60,45); title("noise sample 2");

dct_im3 = dct2(im3_noise);
s = size(im3_noise);
sigma3 = dct_im3(floor((s(1)*0.93)):end,floor((s(2)*0.93)):end).*dct_im3(floor((s(1)*0.93)):end,floor((s(2)*0.93)):end);
sigmasize = size(sigma3);
xs = [1:sigmasize(1)];ys = [1:sigmasize(2)];
[xs,ys] = meshgrid(ys,xs);
figure; surface(xs,ys,sigma3); view(60,45); title("noise sample 3");

dct_im4 = dct2(im4_noise);
s = size(im4_noise);
sigma4 = dct_im4(floor((s(1)*0.9)):end,floor((s(2)*0.9)):end).*dct_im4(floor((s(1)*0.9)):end,floor((s(2)*0.9)):end);
sigmasize = size(sigma4);
xs = [1:sigmasize(1)];ys = [1:sigmasize(2)];
[xs,ys] = meshgrid(ys,xs);
figure; surface(xs,ys,sigma4); view(60,45); title("noise sample 4");

noise_var1 = mean(mean(sigma1));
noise_var2 = mean(mean(sigma2));
noise_var3 = mean(mean(sigma3));
noise_var4 = mean(mean(sigma4));

%5
beta = 3.0;
nose_var1 = beta*noise_var1;
nose_var2 = beta*noise_var2;
nose_var3 = beta*noise_var3;
nose_var4 = beta*noise_var4;

signal_var1 = dct_im1.*dct_im1 + 0.001;
wiener_filter1 = 1 + (noise_var1./signal_var1);
wiener_filter1 = 1./wiener_filter1;
filtered_dct1 = dct_im1.*wiener_filter1;
filtered_dct_image1 = idct2(filtered_dct1);
filtered_dct_image1 = im2uint8(filtered_dct_image1);
imshowpair(im1_noise, filtered_dct_image1, "montage"); title("Noisy Image(Left), DCT Wiener Filtered(Right)");

signal_var2 = dct_im2.*dct_im2 + 0.001;
wiener_filter2 = 1 + (noise_var2./signal_var2);
wiener_filter2 = 1./wiener_filter2;
filtered_dct2 = dct_im2.*wiener_filter2;
filtered_dct_image2 = idct2(filtered_dct2);
filtered_dct_image2 = im2uint8(filtered_dct_image2);
imshowpair(im2_noise, filtered_dct_image2, "montage"); title("Noisy Image(Left), DCT Wiener Filtered(Right)");

signal_var3 = dct_im3.*dct_im3 + 0.001;
wiener_filter3 = 1 + (noise_var3./signal_var3);
wiener_filter3 = 1./wiener_filter3;
filtered_dct3 = dct_im3.*wiener_filter3;
filtered_dct_image3 = idct2(filtered_dct3);
filtered_dct_image3 = im2uint8(filtered_dct_image3);
imshowpair(im3_noise, filtered_dct_image3, "montage"); title("Noisy Image(Left), DCT Wiener Filtered(Right)");

signal_var4 = dct_im4.*dct_im4 + 0.001;
wiener_filter4 = 1 + (noise_var4./signal_var4);
wiener_filter4 = 1./wiener_filter4;
filtered_dct4 = dct_im4.*wiener_filter4;
filtered_dct_image4 = idct2(filtered_dct4);
filtered_dct_image4 = im2uint8(filtered_dct_image4);
imshowpair(im4_noise, filtered_dct_image4, "montage"); title("Noisy Image(Left), DCT Wiener Filtered(Right)");

%6
roberts1 = edge(im1_ycomp, "roberts");
prewitt1 = edge(im1_ycomp, "prewitt");
sobel1 = edge(im1_ycomp, "sobel");
canny1_def = edge(im1_ycomp, "canny"); 
canny1 = edge(im1_ycomp, "canny", [0.01,0.25]); 
subplot(231); imshow(im1_ycomp); title("original");
subplot(232); imshow(roberts1); title("roberts");
subplot(233); imshow(prewitt1); title("prewitt");
subplot(234); imshow(sobel1); title("sobel");
subplot(235); imshow(canny1_def); title("canny");
subplot(236); imshow(canny1); title("canny after smooth");

roberts2 = edge(im2_ycomp, "roberts");
prewitt2 = edge(im2_ycomp, "prewitt");
sobel2 = edge(im2_ycomp, "sobel");
canny2_def = edge(im2_ycomp, "canny");
canny2 = edge(im2_ycomp, "canny", [0.03, 0.3]);
subplot(231); imshow(im2_ycomp); title("original");
subplot(232); imshow(roberts2); title("roberts");
subplot(233); imshow(prewitt2); title("prewitt");
subplot(234); imshow(sobel2); title("sobel");
subplot(235); imshow(canny2_def); title("canny");
subplot(236); imshow(canny2); title("canny after smooth");

roberts3 = edge(im3_ycomp, "roberts");
prewitt3 = edge(im3_ycomp, "prewitt");
sobel3 = edge(im3_ycomp, "sobel");
canny3_def =  edge(im3_ycomp, "canny");
canny3 = edge(im3_ycomp, "canny", [0.08,0.1]);
subplot(231); imshow(im2uint8(im3_ycomp)); title("original");
subplot(232); imshow(roberts3); title("roberts");
subplot(233); imshow(prewitt3); title("prewitt");
subplot(234); imshow(sobel3); title("sobel");
subplot(235); imshow(canny3_def); title("canny");
subplot(236); imshow(canny3); title("canny after smooth");

roberts4 = edge(im4_ycomp, "roberts");
prewitt4 = edge(im4_ycomp, "prewitt");
sobel4 = edge(im4_ycomp, "sobel");
canny4_def = edge(im4_ycomp, "canny");
canny4 = edge(im4_ycomp, "canny",[0.05, 0.35]);
subplot(231); imshow(im4_ycomp); title("original");
subplot(232); imshow(roberts4); title("roberts");
subplot(233); imshow(prewitt4); title("prewitt");
subplot(234); imshow(sobel4); title("sobel");
subplot(235); imshow(canny4_def); title("canny");
subplot(236); imshow(canny4); title("canny after smooth");

%8
[H,T,R] = hough(canny3);

P  = houghpeaks(H,5);
imshow(H,[],'XData',T,'YData',R,'InitialMagnification','fit');
xlabel('\theta'), ylabel('\rho');
axis on, axis normal, hold on;
plot(T(P(:,2)),R(P(:,1)),'s','color','white');

lines = houghlines(canny3,T,R,P);
figure, imshow(canny3), hold on
for k = 1:length(lines)
   xy = [lines(k).point1; lines(k).point2];
   plot(xy(:,1),xy(:,2),'LineWidth',2,'Color',"green");
end
%***********************************
[H,T,R] = hough(canny4);

P  = houghpeaks(H,5);
imshow(H,[],'XData',T,'YData',R,'InitialMagnification','fit');
xlabel('\theta'), ylabel('\rho');
axis on, axis normal, hold on;
plot(T(P(:,2)),R(P(:,1)),'s','color','white');

lines = houghlines(canny4,T,R,P);
figure, imshow(canny4), hold on
for k = 1:length(lines)
   xy = [lines(k).point1; lines(k).point2];
   plot(xy(:,1),xy(:,2),'LineWidth',2,'Color',"green");
end

%9
%9.a&b
im5 = im2double(imread("friend.png"));
imshow(im2uint8(im5));
im5_ycbcr = rgb2ycbcr(im5);
im5_ycomp = im5_ycbcr(:,:,1);
im5_cbcomp = im5_ycbcr(:,:,2);
im5_crcomp = im5_ycbcr(:,:,3);
roberts5_y = edge(im5_ycomp,"roberts");
prewitt5_y = edge(im5_ycomp,"prewitt");
sobel5_y = edge(im5_ycomp,"sobel");
canny5_y = edge(im5_ycomp,"canny");
%9.c
subplot(231); imshow(im2uint8(im5)); title("original");
subplot(232); imshow(roberts5_y); title("roberts y");
subplot(233); imshow(prewitt5_y); title("prewitt y");
subplot(235); imshow(sobel5_y); title("sobel y");
subplot(236); imshow(canny5_y); title("canny y");

roberts5_cb = edge(im5_cbcomp,"roberts");
prewitt5_cb = edge(im5_cbcomp,"prewitt");
sobel5_cb = edge(im5_cbcomp,"sobel");
canny5_cb = edge(im5_cbcomp,"canny");
subplot(231); imshow(im2uint8(im5)); title("original");
subplot(232); imshow(roberts5_cb); title("roberts cb");
subplot(233); imshow(prewitt5_cb); title("prewitt cb");
subplot(235); imshow(sobel5_cb); title("sobel cb");
subplot(236); imshow(canny5_cb); title("canny cb");

roberts5_cr = edge(im5_crcomp,"roberts");
prewitt5_cr = edge(im5_crcomp,"prewitt");
sobel5_cr = edge(im5_crcomp,"sobel");
canny5_cr = edge(im5_crcomp,"canny");
subplot(231); imshow(im2uint8(im5)); title("original");
subplot(232); imshow(roberts5_cr); title("roberts cr");
subplot(233); imshow(prewitt5_cr); title("prewitt cr");
subplot(235); imshow(sobel5_cr); title("sobel cr");
subplot(236); imshow(canny5_cr); title("canny cr");
%9.d
X = im5(:,:,2);
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
V3img = wcodemat(V3,255,'mat',1);
H3img = wcodemat(H3,255,'mat',1);
D3img = wcodemat(D3,255,'mat',1);
A3img = wcodemat(A3,255,'mat',1);
[H4,V4,D4] = detcoef2('all',c,s,4);
A4 = appcoef2(c,s,'haar',4); 
V4img = wcodemat(V4,255,'mat',1);
H4img = wcodemat(H4,255,'mat',1);
D4img = wcodemat(D4,255,'mat',1);
A4img = wcodemat(A4,255,'mat',1);

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

%10
%10.a
lena =  imread('Lena.png'); 
lena_ycbcr = rgb2ycbcr(lena);
lena_ycomp = lena_ycbcr(:,:,1);

[c,s]=wavedec2(lena_ycomp,4,'haar');
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
V3img = wcodemat(V3,255,'mat',1);
H3img = wcodemat(H3,255,'mat',1);
D3img = wcodemat(D3,255,'mat',1);
A3img = wcodemat(A3,255,'mat',1);

[H4,V4,D4] = detcoef2('all',c,s,4);
A4 = appcoef2(c,s,'haar',4); 
V4img = wcodemat(V4,255,'mat',1);
H4img = wcodemat(H4,255,'mat',1);
D4img = wcodemat(D4,255,'mat',1);
A4img = wcodemat(A4,255,'mat',1);


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
%10.b
size1 = size(V1img,1);
size2 = size(V1img,2);
absim1 = zeros(size1,size2);
for i = 1:size1
    for j = 1:size2
        absim1(i,j) = sqrt(V1img(i,j)*V1img(i,j)+H1img(i,j)*H1img(i,j));
    end
end
figure;
absim1 = im2uint8(absim1/255);
imshow(absim1);

size1 = size(V2img,1);
size2 = size(V2img,2);
absim2 = zeros(size1,size2);
for i = 1:size1
    for j = 1:size2
        absim2(i,j) = sqrt(V2img(i,j)*V2img(i,j)+H2img(i,j)*H2img(i,j));
    end
end
figure;
absim2 = im2uint8(absim2/255);
imshow(absim2);

size1 = size(V3img,1);
size2 = size(V3img,2);
absim3 = zeros(size1,size2);
for i = 1:size1
    for j = 1:size2
        absim3(i,j) = sqrt(V3img(i,j)*V3img(i,j)+H3img(i,j)*H3img(i,j));
    end
end
figure;
absim3 = im2uint8(absim3/255);
imshow(absim3);

size1 = size(V4img,1);
size2 = size(V4img,2);
absim4 = zeros(size1,size2);
for i = 1:size1
    for j = 1:size2
        absim4(i,j) = sqrt(V4img(i,j)*V4img(i,j)+H4img(i,j)*H4img(i,j));
    end
end
figure;
absim4 = im2uint8(absim4/255);
imshow(absim4);
%10.c&d
Lenaycbcrwed = imread("LenaYCbCrWED.png");
Lenaycbcrwed_ycbcr = rgb2ycbcr(Lenaycbcrwed);
Lenaycbcrwed_ycomp = Lenaycbcrwed_ycbcr(:,:,1);
subplot(121); imshow(absim1); title("step b");
subplot(122); imshow(Lenaycbcrwed_ycomp); title("step c");
%10.e
T = graythresh(absim1);
absim1_bi = imbinarize(absim1,T);
Lenaycbcrwed_ycomp_bi = imbinarize(Lenaycbcrwed_ycomp,0.4);
subplot(121); imshow(absim1_bi); title("step b (T = 0.1843)");
subplot(122); imshow(Lenaycbcrwed_ycomp_bi); title("step c (T = 0.4)");




























        



