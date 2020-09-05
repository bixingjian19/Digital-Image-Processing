function [LL, LH, HL, HH] = haar_dwt2D(img)
[m, n] = size(img);
for i = 1:m
    [L, H] = haar_dwt(img(i,:));
    img(i,:) = [L, H];
end
for j = 1:n
    [L, H] = haar_dwt(img(:,j));
    img(:,j) = [L, H];
end
LL = mat2gray(img(1:m/2,1:n/2));
LH = mat2gray(img(1:m/2,n/2+1:n));
HL = mat2gray(img(m/2+1:n,1:n/2));
HH = mat2gray(img(m/2+1:n,n/2+1:n));
end

