%1
im1 = imread("us1.jpg");%ultrasound1
im2 = imread("us2.jpg");%ultrasound2
im3 = imread("mg1.jpg");%mammogram1
im4 = imread("mg2.jpg");%mammogram2
subplot(1,4,1); imshow(im1); title("im1");
subplot(1,4,2); imshow(im2); title("im2");
subplot(1,4,3); imshow(im3); title("im3");
subplot(1,4,4); imshow(im4); title("im4");

%2
%conver color image to gray scale image
bw1 = rgb2gray(im1);
bw2 = rgb2gray(im2);
bw3 = rgb2gray(im3);
bw4 = rgb2gray(im4);
size_of_bw1 = size(bw1);

%3
bw1_adj1 = imadjust(bw1,[],[],0.45); 
bw1_adj2 = imadjust(bw1,[],[],2.2);
subplot(1,3,1); imshow(bw1_adj1); title("gamma:0.45");
subplot(1,3,2); imshow(bw1); title("original");
subplot(1,3,3); imshow(bw1_adj2); title("gamma:2.2");

%4
bw3_invert =  imadjust(bw3,[0,1],[1,0]);
subplot(1,2,1); imshow(bw3); title("original");
subplot(1,2,2); imshow(bw3_invert); title("bw3_invert")

%1
bw1_mat2gray = mat2gray(bw1);
imshow(bw1_mat2gray);

%2
bw1_im2uint8 = im2uint8(bw1_mat2gray);

%3
im_city = imread("city.jpg");
gray_im = rgb2gray(im_city);
double_value = im2double(gray_im);
G = 800*log(1+double_value);
Gs = im2uint8(mat2gray(G));
subplot(1,2,2); imshow(Gs);
subplot(1,2,1); imshow(gray_im);

%4
im_resize1 = imresize(im_city,3);
im_resize2 = imresize(im_city,0.5);
size1 = size(im_city);
size2 = size(im_resize1);
size3 = size(im_resize2);





