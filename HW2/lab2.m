%2.1
%1
im1 = imread("pout.tif");
imhist(im1);
%2
gaussian = imnoise(im1,"gaussian");
imshow(gaussian);
%3%4
median_filtering = medfilt2(gaussian);
h = fspecial("average");
smoothing_filtering = imfilter(gaussian,h,"replicate");
imshowpair(median_filtering, smoothing_filtering,"montage");
%6
im2 = imread("cameraman.tif");
h = fspecial('laplacian'); 
laplacian = imfilter(im2,h,'replicate');
imshow(laplacian)
%design
kernel1 = [1,2,3;1,2,3;1,2,3]/9;
test1 = imfilter(gaussian,kernel1);
subplot(141); imshow(gaussian); title("original");
subplot(142); imshow(median_filtering); title("median filter");
subplot(143); imshow(smoothing_filtering); title("smoothing filter");
subplot(144); imshow(test1); title("mine kernel")


%2.2
%1
h = fspecial("average");
Average = imfilter(gaussian,h,"replicate");
imshowpair(gaussian,Average,"montage"); 
%2
time = zeros(30);
for i = 1:2:30
    h = fspecial("average",i);
    tic
    Average = imfilter(gaussian,h,"replicate");
    toc
    time(i) = toc;
end
plot(1:2:30,time(1:2:30));
%3
time1 = zeros(20);
for i = 1:2:20
    tic
        K2=medfilt2(gaussian,[i i]);
    toc
    time1(i) = toc;
end
plot(1:2:20,time1(1:2:20));


%2.3
H = fspecial('sobel');
Sobel = imfilter(im1,H,'replicate');
Sobel1 = imfilter(im1,H','replicate');
subplot(121); imshow(Sobel); title("sober")
subplot(122); imshow(Sobel1); title("sober'")

H = fspecial('prewitt');
Prewitt = imfilter(im1,H,'replicate');
Prewitt1 = imfilter(im1,H','replicate');
subplot(121); imshow(Prewitt); title("prewitt");
subplot(122); imshow(Prewitt1); title("prewitt'");


%2.4
%1
im1 = imread("pout.tif");
im1noise = imnoise(im1,"gaussian",0,0.1);
imshow(im1noise);
%2
[size1, size2] = size(im1); 
tempmat = zeros(size1,size2);
times = 10;
for i = 1:times
    tempnoise = im2double(imnoise(im1,"gaussian",0,0.1));
    tempmat = tempmat + tempnoise;
end
tempmat = tempmat/times;
subplot(151); imshow(im1noise); title("original")
subplot(152); imshow(tempmat); title("10 times")
%3
subplot(153); imshow(noiseiterations(50)); title("50 times");
subplot(154); imshow(noiseiterations(100)); title("100 times");
subplot(155); imshow(noiseiterations(1000)); title("1000 times");

    
    