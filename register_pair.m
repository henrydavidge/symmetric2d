function [f_c,f_r,img2_o_f] = register_pair(img1,img2,rho,lambda,maxIter)

%If inputs are images, read them
if (ischar(img1))
    img1 = double(imread(img1));
    if (size(img1) == 3)
        img1 = (img1(:,:,1) + img1(:,:,2) + img1(:,:,3))/3;
    end
end

if (ischar(img2))
    img2 = double(imread(img2));
    if (size(img1) == 3)
        img2 = (img2(:,:,1) + img2(:,:,2) + img2(:,:,3))/3;
    end
end

imwrite(mat2gray(img1), 'out/img1.png', 'png');
imwrite(mat2gray(img2), 'out/img2.png', 'png');
imwrite(mat2gray(img1-img2), 'out/diff_orig.png', 'png');

imgSize = size(img1);
rows = imgSize(1);
cols = imgSize(2);

%Register the images
%Initialize the registration function
[f_c,f_r]=meshgrid(1:cols,1:rows); %creates lookup tables

[dr_img2_orig,dc_img2_orig]=grad_img(img2);

%Call the registration function 
tic;

%Calculate grid size. Blocks are fixed at 32x32.
grid = [ceil(cols/32), ceil(rows/32)];
   
%Initialize kernels
interp = parallel.gpu.CUDAKernel('interp2.ptx','interp2.cu');
interp.GridSize = grid;
interp.ThreadBlockSize = [32,32];

extf = parallel.gpu.CUDAKernel('register2d.ptx', 'register2d.cu', 'extf');
extf.GridSize = grid;
extf.ThreadBlockSize = [32,32];

intf = parallel.gpu.CUDAKernel('register2d.ptx', 'register2d.cu', 'intf');
intf.GridSize = grid;
intf.ThreadBlockSize = [32,32];

add = parallel.gpu.CUDAKernel('register2d.ptx', 'register2d.cu', 'add');
add.GridSize = grid;
add.ThreadBlockSize = [32,32];

d_f = parallel.gpu.CUDAKernel('register2d.ptx', 'register2d.cu', 'd_f');
d_f.GridSize = grid;
d_f.ThreadBlockSize = [32,32];

%Initialize variables on the GPU

dr_img2 = gpuArray(dr_img2_orig);
dc_img2 = gpuArray(dc_img2_orig);

img2_o_f = gpuArray(img2);


f_c = gpuArray(f_c);
f_r = gpuArray(f_r);

img1 = gpuArray(img1);
img2 = gpuArray(img2);

extf_r = parallel.gpu.GPUArray.zeros(size(f_c));
extf_c = parallel.gpu.GPUArray.zeros(size(f_c));

intf_r = parallel.gpu.GPUArray.zeros(size(f_c));
intf_c = parallel.gpu.GPUArray.zeros(size(f_c));

d_f_r = parallel.gpu.GPUArray.zeros(size(f_c));
d_f_c = parallel.gpu.GPUArray.zeros(size(f_c));


for k=1:maxIter,
    %Interpolate image2
    img2_o_f = feval(interp, img2_o_f, f_c, f_r, img2, rows, cols);
    
    dr_img2 = feval(interp, dr_img2, f_c, f_r, dr_img2_orig, rows, cols);
    dc_img2 = feval(interp, dc_img2, f_c, f_r, dc_img2_orig, rows, cols);

    %external energies
    extf_r = feval(extf, extf_r, img1, img2_o_f, dr_img2, rows, cols); 
    extf_c = feval(extf, extf_c, img1, img2_o_f, dc_img2, rows, cols);
    
    %Internal energies
    intf_r = feval(intf, intf_r, f_r, rows, cols);
    intf_c = feval(intf, intf_c, f_c, rows, cols);
    
    %Update Images
    d_f_r = feval(d_f, d_f_r, intf_r, extf_r, rho, lambda, rows, cols);
    d_f_r = min(max(d_f_r,-3),3);
    f_r = feval(add, f_r, f_r, d_f_r, rows, cols);
    
    f_r = max(min(f_r, rows),1);
    
    d_f_c = feval(d_f, d_f_c, intf_c, extf_c, rho, lambda, rows, cols);
    d_f_c = min(max(d_f_c,-3),3);
    f_c = feval(add, f_c, f_c, d_f_c, rows, cols);
    
    f_c = max(min(f_c, cols),1);
    
end
img1 = gather(img1);
img2_o_f = gather(img2_o_f);

imwrite(mat2gray(img2_o_f), 'out/interpolated2.png', 'png');
imwrite(mat2gray(img1 - img2_o_f), 'out/diff.png', 'png');
disp(sum((img1(:) - img2_o_f(:)).^2));
toc;

function plot_profile(img1,img2,row)
hold off;
tmp=img1(row,:);
plot(tmp,'b');
hold on;
tmp=img2(row,:);
plot(tmp,'r');

function img_out=filter_image(img,sigma)
h=fspecial('gaussian',13,sigma);
img_out=conv2(img,h,'same');

function [dr_img,dc_img]=grad_img(img)
dr_img=zeros(size(img));
dc_img=zeros(size(img));

dr_img(2:end-1,2:end-1)=img(3:end,2:end-1)-img(1:end-2,2:end-1); %why is it offset by 2 and not 1?
dc_img(2:end-1,2:end-1)=img(2:end-1,3:end)-img(2:end-1,1:end-2);