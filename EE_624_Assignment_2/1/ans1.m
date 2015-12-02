[img,map]= imread('input_text.png');
newmap=rgb2gray(map);
[m,n]=size(img);

%Creating binary image
newimage=zeros(m,n);
for i=1:m
    for j=1:n
        if(img(i,j)~=15)
            newimage(i,j)=1;
        end
    end
end

imshow(newimage)

se=strel('line',6,0);
im2=imdilate(newimage,se);
figure;
imshow(im2);

%A pixel of each of the connected components
point_connected_component=[114,27; 174,25; 241,29; 341,26; 109,64; 184,64; 234,63; 283,66; 338,67; 89,110; 164,106; 246,104; 346,108; 71,147; 136,141; 219,141; 285,144; 373,143];
[num_components,junk]=size(point_connected_component);

%Extracting the connected components
se=strel('square',3);
for i=1:num_components
    x=zeros(m,n);
    y=zeros(m,n);
    x(point_connected_component(i,2),point_connected_component(i,1))=1;
    while(min(min(x(:,:)==y(:,:)))~=1)
        y=x;
        z=imdilate(x,se);
        x=z & im2;
    end
imwrite(x&newimage,strcat('component',int2str(i),'.jpg'));
end