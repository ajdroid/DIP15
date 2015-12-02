original= im2double(imread('Lenna.noise.jpg'));

[m,n] = size(original);
T = 50; 
A = (zeros(m,n,T)); 
I = (zeros(m,n,T));
A(:,:,1) = original;
I(:,:,1) = original;

Gx =  [-1 0 1; -2 0 2; -1 0 1];
Gy =  [-1 -2 -1; 0 0 0 ; 1 2 1];

K=3;
lamda = 0.1;
c  = im2double(zeros(m,n,T-1));
TGrad = zeros(m,n);
N = zeros(m,n);
S = zeros(m,n);
E = zeros(m,n);
W = zeros(m,n);
N_iso = zeros(m,n);

S_iso = zeros(m,n);

E_iso = zeros(m,n);

W_iso = zeros(m,n);


for t= 1:T-1
M1 = zeros(m,n);
M2 = M1;
M3 = M1;
M4 = M1;
M5 = M1;
M6 = M1;


M1(2:m,2:n) = -1*A(1:m-1,1:n-1,t);
M2(1:m,2:n) = -2*A(1:m,1:n-1,t);
M3(1:m-1,2:n) = -1*A(2:m,1:n-1,t);
M4(2:m,1:n-1) = 1*A(1:m-1,2:n,t);
M5(1:m,1:n-1) = 2*A(1:m,2:n,t);
M6(1:m-1,1:n-1) = 1*A(2:m,2:n,t);
XGrad = M1+M2+M3+M4+M5+M6;


M1(2:m,2:n) = -1*A(1:m-1,1:n-1,t);

M2(2:m,1:n)  = -2*A(1:m-1,1:n,t);
M3(1:m-1,2:n) = -1*A(2:m,1:n-1,t);
M4(1:m-1,2:n) = 1*A(2:m,1:n-1,t);
M5(1:m-1,1:n) = 2*A(2:m,1:n,t);
M6(1:m-1,1:n-1) = 1*A(2:m,2:n,t);
YGrad = M1+M2+M3+M4+M5+M6;

 TGrad = abs(XGrad) + abs(YGrad);
 c(:,:,t) = exp( -(TGrad.*TGrad)/K );
 N = zeros(m,n);
 N(1:m-1,:) = (A(2:m,:,t) -A(1:m-1,:,t) ).*c(2:m,:,t) ;
 N_iso(1:m-1,:) = (I(2:m,:,t) -I(1:m-1,:,t) );
 S = zeros(m,n);
 S(2:m,:) = (A(2:m,:,t) -A(1:m-1,:,t) ).*c(1:m-1,:,t) ;
 S_iso(2:m,:) = (I(2:m,:,t) -I(1:m-1,:,t) ) ;
 
 E = zeros(m,n);
 E(:,1:n-1) = (A(:,2:n,t) -A(:,1:n-1,t) ).*c(:,2:n,t) ;
 E_iso(:,1:n-1) = (I(:,2:n,t) -I(:,1:n-1,t) ) ;
 W = zeros(m,n);
 W(:,2:n) = (A(:,2:n,t) -A(:,1:n-1,t) ).*c(:,1:n-1,t) ;
 W_iso(:,2:n) = (I(:,2:n,t) -I(:,1:n-1,t) );
 
 A(:,:,t+1) = A(:,:,t) + lamda*0.35*(N-S + E -W);
 I(:,:,t+1) = I(:,:,t) + lamda*(N_iso - S_iso + E_iso - W_iso);

end

figure
imshow(original);
title('original');

figure
imshow((I(:,:,T)));
title('Isotropic output');

figure
imshow((A(:,:,T)));
title('Anisotropic output');


%part c

N_local_mean= im2double(imread('Lenna.noise.jpg'));

[m,n] = size(N_local_mean);
img = zeros(m+6,n+6);
new_img = zeros(m,n);

img(4:m+3, 4:n+3) = N_local_mean;

W_size = 2; 
W_weight = zeros(2*W_size+1,2*W_size+1);
Nrom_weight = 0;
Pix_val = 0;
P_size = 3; 
h = sqrt(1);

for i = 4:m+3
    for j = 4:n+3
  
 W_weight = zeros(5,5);
  Norm_weight = 0;
  Pix_val = 0;
for wi = -W_size:+W_size
for wj = -W_size:+W_size
               
 if( i+wi>=4 && i+wi<=m+3 && j+wj>+4 && j+wj<=n+3)
 original = img( i+wi-P_size:i+wi+P_size,j+wj-P_size:j+wj+P_size) - img( i-P_size:i+P_size,j-P_size:j+P_size);
  W_weight(W_size+1+wi,W_size+1+wj) = exp( -((norm(original))^2)/(h^2) ) ;
                       
  Norm_weight = Norm_weight + exp( -((norm(original))^2)/(h^2) ) ;
   Pix_val = Pix_val + img( i+wi,j+wj)*W_weight(W_size+1+wi,W_size+1+wj);
end
 end
  end
new_img( i-3,j-3) =  Pix_val/Norm_weight ;
 end
end

figure
imshow(new_img);
title('Non-local-means output');


                    
