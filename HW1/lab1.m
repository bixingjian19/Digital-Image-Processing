%Task1
%1
pout = imread("pout.tif");

%2
figure();
imagesc(pout);


%3 imadjust(pout,[50/255,150/255,[0,1]) will not give the correct result
%because pixels whose intensity is between 50 and 150 will change, so we have to traverse this
%image.
for i = 1:size(pout,1)
    for j = 1:size(pout,2)
        if pout(i,j) < 50
            pout(i,j) = 0;
        end
        if pout(i,j) > 150
            pout(i,j) = 255;
        end
    end
end

%Task2
pout = imread("pout.tif");
pout_flipped = flip(pout,2);
pout_flipped2 = flip(pout_flipped,1);
imshow(pout_flipped2);

