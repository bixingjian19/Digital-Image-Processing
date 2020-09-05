function output = scalebymax(image)
%This function is to do scale-by-max operation.
[w,h,d] = size(image);
max = 0;
for i = 1:w
    for j = 1:h
        val = image(i,j,1)+image(i,j,2)+image(i,j,3);
        if val > max
            max = val;
            maxi = i;
            maxj = j;
        end
    end
end
output(:,:,1) = image(:,:,1)*(255/image(maxi,maxj,1));
output(:,:,2) = image(:,:,2)*(255/image(maxi,maxj,2));
output(:,:,3) = image(:,:,3)*(255/image(maxi,maxj,3));
end
