img = imread('unenhanced.tif');

% Get the dimensions of the image.  
[rows, columns] = size(img);

initial_min = 1.0; initial_max=0.0;
img = double(img)/255.0;
for i=1:rows
    for j=1:columns
        initial_min=min(img(i,j),initial_min);
        initial_max=max(img(i,j),initial_max);
    end
end


%%2(a)
enhanced_img = img;
for i=1:rows
    for j=1:columns
        enhanced_img(i,j) = 1.0*(img(i,j)- initial_min)/double(initial_max - initial_min);
    end
end

figure(1);
%Display the original gray scale image.
subplot(3, 2, 1);
imshow(uint8(255*img));
title('Original Image');

% Histogram of originaal image.

subplot(3, 2, 2); 
imhist(uint8(255*img));
title('Histogram of Original Image');

subplot(3, 2, 3);
imshow(uint8(255*enhanced_img));
title('Contrast streched Image');

% Histogram of enhanced image.
subplot(3, 2, 4); 
imhist(uint8(255*enhanced_img));
title('Histogram of Contrast streched Image');


%%2(b)



enhanced_img1=enhanced_img;
for i=1:rows
    for j=1:columns
        enhanced_img1(i,j) = 3.5*img(i,j)^(3);
    end
end



subplot(3, 2, 5);
imshow(uint8(255*enhanced_img1));
title('Power law transformed Image');

%Histogram of enhanced image.
subplot(3, 2, 6); 
imhist(uint8(255*enhanced_img1));
title('Histogram of Power law Transformed Image');



% 2(c)

img = imread('unenhanced.tif');
[rows, columns] = size(img);
freq = zeros(1,256);
freqSum=freq;
map=freqSum;
enhanced_img2=img;
for i=1:rows
    for j=1:columns
        freq(img(i,j)+1)= freq(img(i,j)+1)+1;
    end
end
for i=2:256
    freqSum(i)=freqSum(i-1)+freq(i);
    map(i)=255*freqSum(i)/(rows*columns);
end
    
for i=1:rows
    for j=1:columns
        enhanced_img2(i,j)= map(img(i,j));
    end
end


figure(2);
imshow(enhanced_img2);
title('Histogram Equalized Image');


