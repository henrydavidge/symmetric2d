function register_pair_symmetric

%Create two test images

[x,y] = meshgrid(-64:63, -64:63);
z = x.^2 + y.^2;
imgSize= 128;
img1=zeros(imgSize,imgSize);
img1(z <= 40^2) = 1;
img1=filter_image(img1,5);
img2=zeros(imgSize,imgSize);
img2(z <= 25^2)=1;
img2=filter_image(img2,5);

imwrite(mat2gray(img1), 'out/img1.png', 'png');
imwrite(mat2gray(img2), 'out/img2.png', 'png');


%Register the images
%Initialize the registration function
[fCol,fRow]=meshgrid(1:imgSize,1:imgSize);
gCol = fCol;
gRow = fRow;
%creates lookup tables
%Call the registration function
tic;
[fc1,fr1,gc1,gr1]=register_images(img2,img1,fCol,fRow,gCol,gRow,.1,4000);
toc;




function [f_c,f_r,g_c,g_r]=register_images(img1,img2,f_c,f_r,g_c,g_r,rho,maxIter)
lambda = .05;
lambda2 = -.4;
lambda3 = 1;
imgSize = 128;

interp = parallel.gpu.CUDAKernel('interp2.ptx','interp2.cu');
interp.GridSize = [4,4];
interp.ThreadBlockSize = [32,32];

jacPartialsAndBarrier = parallel.gpu.CUDAKernel('symmetric2d.ptx', 'symmetric2d.cu', 'jacPartialsAndBarrier');
jacPartialsAndBarrier.GridSize = [4,4];
jacPartialsAndBarrier.ThreadBlockSize = [32,32];

jacobian = parallel.gpu.CUDAKernel('symmetric2d.ptx', 'symmetric2d.cu', 'jacobian');
jacobian.GridSize = [4,4];
jacobian.ThreadBlockSize = [32,32];

extf = parallel.gpu.CUDAKernel('symmetric2d.ptx', 'symmetric2d.cu', 'extf');
extf.GridSize = [4,4];
extf.ThreadBlockSize = [32,32];

intf = parallel.gpu.CUDAKernel('symmetric2d.ptx', 'symmetric2d.cu', 'intf');
intf.GridSize = [4,4];
intf.ThreadBlockSize = [32,32];

add = parallel.gpu.CUDAKernel('symmetric2d.ptx', 'symmetric2d.cu', 'add');
add.GridSize = [4,4];
add.ThreadBlockSize = [32,32];

d_f = parallel.gpu.CUDAKernel('symmetric2d.ptx', 'symmetric2d.cu', 'd_f');
d_f.GridSize = [4,4];
d_f.ThreadBlockSize = [32,32];


[dr_img2_orig,dc_img2_orig]=grad_img(img2);
[dr_img1_orig,dc_img1_orig]=grad_img(img1);

dr_img2 = gpuArray(dr_img2_orig);
dc_img2 = gpuArray(dc_img2_orig);
dr_img1 = gpuArray(dr_img2_orig);
dc_img1 = gpuArray(dc_img2_orig);

img2_o_f = gpuArray(img2);
img1_o_g = gpuArray(img1);
jacf = parallel.gpu.GPUArray.ones(size(f_c));
jacg = parallel.gpu.GPUArray.ones(size(f_c));
jacf_temp = jacf;
jacg_temp = jacg;


f_c = gpuArray(f_c);
f_r = gpuArray(f_r);
g_c = gpuArray(g_c);
g_r = gpuArray(g_r);
img1 = gpuArray(img1);
img2 = gpuArray(img2);

extf_r = parallel.gpu.GPUArray.zeros(size(f_c));
extf_c = parallel.gpu.GPUArray.zeros(size(f_c));

intf_r = parallel.gpu.GPUArray.zeros(size(f_c));
intf_c = parallel.gpu.GPUArray.zeros(size(f_c));

f_c_temp = f_c;
f_r_temp = f_r;

d_f_r = parallel.gpu.GPUArray.zeros(size(f_c));
d_f_c = parallel.gpu.GPUArray.zeros(size(f_c));

fi_m_1x = parallel.gpu.GPUArray.ones(size(f_c));
fi_p_1x = parallel.gpu.GPUArray.ones(size(f_c));
fj_m_1x = parallel.gpu.GPUArray.ones(size(f_c));
fj_p_1x = parallel.gpu.GPUArray.ones(size(f_c));

fi_m_1y = parallel.gpu.GPUArray.ones(size(f_c));
fi_p_1y = parallel.gpu.GPUArray.ones(size(f_c));
fj_m_1y = parallel.gpu.GPUArray.ones(size(f_c));
fj_p_1y = parallel.gpu.GPUArray.ones(size(f_c));

fbarrierx = parallel.gpu.GPUArray.zeros(size(f_c));

fbarriery = parallel.gpu.GPUArray.zeros(size(f_c));

g_c = gpuArray(f_c);
g_r = gpuArray(f_r);

extg_r = parallel.gpu.GPUArray.zeros(size(f_c));
extg_c = parallel.gpu.GPUArray.zeros(size(f_c));

intg_r = parallel.gpu.GPUArray.zeros(size(f_c));
intg_c = parallel.gpu.GPUArray.zeros(size(f_c));

d_g_r = parallel.gpu.GPUArray.zeros(size(f_c));
d_g_c = parallel.gpu.GPUArray.zeros(size(f_c));

gi_m_1x = parallel.gpu.GPUArray.ones(size(f_c));
gi_p_1x = parallel.gpu.GPUArray.ones(size(f_c));
gj_m_1x = parallel.gpu.GPUArray.ones(size(f_c));
gj_p_1x = parallel.gpu.GPUArray.ones(size(f_c));

gi_m_1y = parallel.gpu.GPUArray.ones(size(f_c));
gi_p_1y = parallel.gpu.GPUArray.ones(size(f_c));
gj_m_1y = parallel.gpu.GPUArray.ones(size(f_c));
gj_p_1y = parallel.gpu.GPUArray.ones(size(f_c));

gbarrierx = parallel.gpu.GPUArray.zeros(size(f_c));

gbarriery = parallel.gpu.GPUArray.zeros(size(f_c));


for k=1:maxIter,
    if k==2
        save('out.mat');
    end
    %Interpolate image2
    img2_o_f = feval(interp, img2_o_f, f_c, f_r, img2, imgSize, imgSize);
    img1_o_g = feval(interp, img1_o_g, g_c, g_r, img1, imgSize, imgSize);
    
    dr_img2 = feval(interp, dr_img2, f_c, f_r, dr_img2_orig, imgSize, imgSize);
    dc_img2 = feval(interp, dc_img2, f_c, f_r, dc_img2_orig, imgSize, imgSize);
    
    dr_img1 = feval(interp, dr_img1, g_c, g_r, dr_img1_orig, imgSize, imgSize);
    dc_img1 = feval(interp, dc_img1, g_c, g_r, dc_img1_orig, imgSize, imgSize);

    
    %external energies
    extf_r = feval(extf, extf_r, img1_o_g, img2_o_f, dr_img2, imgSize, imgSize);
    extf_c = feval(extf, extf_c, img1_o_g, img2_o_f, dc_img2, imgSize, imgSize);
    
    extg_r = -feval(extf, extg_r, img1_o_g, img2_o_f, dr_img1, imgSize, imgSize);
    extg_c = -feval(extf, extg_c, img1_o_g, img2_o_f, dc_img1, imgSize, imgSize);
    
    
    
    %breaking squared error*partial of jacobian into four terms so I don't
    %go crazy
    [fi_m_1x, fi_p_1x, fj_m_1x, fj_p_1x, fbarrierx] = feval(jacPartialsAndBarrier, fi_m_1x, fi_p_1x, fj_m_1x, fj_p_1x, fbarrierx, jacf, f_r, img1_o_g, img2_o_f, imgSize, imgSize, 1);
    [gi_m_1x, gi_p_1x, gj_m_1x, gj_p_1x, gbarrierx] = feval(jacPartialsAndBarrier, gi_m_1x, gi_p_1x, gj_m_1x, gj_p_1x, gbarrierx, jacg, g_r, img1_o_g, img2_o_f, imgSize, imgSize, 1); 
    
%     barrier/smoothing term - to be used instead of laplacian
%     barrierx = padarray(ax(2:end-1,2:end-1) .* (1 + log(jac(2:end-1,1:end-2)) - 1./jac(2:end-1,1:end-2)) +...
%         bx(2:end-1,2:end-1) .* (1 + log(jac(1:end-2,2:end-1)) - 1./jac(1:end-2,2:end-1)) +...
%         cx(2:end-1,2:end-1) .* (1 + log(jac(2:end-1,3:end)) - 1./jac(2:end-1,3:end)) +...
%         dx(2:end-1,2:end-1) .* (1 + log(jac(3:end,2:end-1)) -
%         1./jac(3:end,2:end-1)), [1,1]);
    
    
    [fi_m_1y, fi_p_1y, fj_m_1y, fj_p_1y, fbarriery] = feval(jacPartialsAndBarrier, fi_m_1y, fi_p_1y, fj_m_1y, fj_p_1y, fbarriery, jacf, f_c, img1_o_g, img2_o_f, imgSize, imgSize, -1);
    [gi_m_1y, gi_p_1y, gj_m_1y, gj_p_1y, gbarriery] = feval(jacPartialsAndBarrier, gi_m_1y, gi_p_1y, gj_m_1y, gj_p_1y, gbarriery, jacg, g_c, img1_o_g, img2_o_f, imgSize, imgSize, -1);
%     barriery = padarray(ay(2:end-1,2:end-1) .* (1 + log(jac(2:end-1,1:end-2)) - 1./jac(2:end-1,1:end-2)) +...
%         by(2:end-1,2:end-1) .* (1 + log(jac(1:end-2,2:end-1)) - 1./jac(1:end-2,2:end-1)) +...
%         cy(2:end-1,2:end-1) .* (1 + log(jac(2:end-1,3:end)) - 1./jac(2:end-1,3:end)) +...
%         dy(2:end-1,2:end-1) .* (1 + log(jac(3:end,2:end-1)) - 1./jac(3:end,2:end-1)), [1,1]);
    
    
   
    
    
    
    
    
    
    
    while (1)
        
%         Internal energies
        f_c_temp = f_c;
        intf_c = feval(intf, intf_c, f_c, imgSize, imgSize);
        d_f_c = feval(d_f, d_f_c, jacf, jacg, fi_m_1x, fj_m_1x, fi_p_1x, fj_p_1x, fbarrierx, intf_c, extf_c, rho, lambda, lambda2, imgSize, imgSize);
        d_f_c=min(max(d_f_c,-3),3); %don't want to jump too much in one iteration
        f_c_temp = feval(add, f_c_temp, f_c, d_f_c, imgSize, imgSize);
        
        f_c_temp=max(min(f_c_temp,imgSize),1); %edge control
        
        f_r_temp = f_r;
        intf_r = feval(intf, intf_r, f_r, imgSize, imgSize);
        d_f_r = feval(d_f, d_f_r, jacf, jacg, fi_m_1y, fj_m_1y, fi_p_1y, fj_p_1y, fbarriery, intf_r, extf_r, rho, lambda, lambda2, imgSize, imgSize);
        d_f_r=min(max(d_f_r,-3),3);
        f_r_temp = feval(add, f_r_temp, f_r, d_f_r, imgSize, imgSize);
        
        f_r_temp=max(min(f_r_temp,imgSize),1);
        
        jacf_temp = feval(jacobian, jacf_temp, f_c_temp, f_r_temp, imgSize, imgSize);
        
        g_c_temp = g_c;
        intg_c = feval(intf, intg_c, g_c, imgSize, imgSize);
        d_g_c = feval(d_f, d_g_c, jacf, jacg, gi_m_1x, gj_m_1x, gi_p_1x, gj_p_1x, gbarrierx, intg_c, extg_c, rho, lambda, lambda2, imgSize, imgSize);
        d_g_c=min(max(d_g_c,-3),3); %don't want to jump too much in one iteration
        g_c_temp = feval(add, g_c_temp, g_c, d_g_c, imgSize, imgSize);
        
        g_c_temp=max(min(g_c_temp,imgSize),1); %edge control
        
        g_r_temp = g_r;
        intg_r = feval(intf, intg_r, g_r, imgSize, imgSize);
        d_g_r = feval(d_f, d_g_r, jacf, jacg, gi_m_1y, gj_m_1y, gi_p_1y, gj_p_1y, gbarriery, intg_r, extg_r, rho, lambda, lambda2, imgSize, imgSize);
        d_g_r=min(max(d_g_r,-3),3);
        g_r_temp = feval(add, g_r_temp, g_r, d_g_r, imgSize, imgSize);
        
        g_r_temp=max(min(g_r_temp,imgSize),1);
        
        jacg_temp = feval(jacobian, jacg_temp, g_c_temp, g_r_temp, imgSize, imgSize);

        
        
%         f_c_temp = f_c;
%         intf_c(2:end-1,2:end-1)= -4*f_c(2:end-1,2:end-1)+ ... %noise reduction?
%             f_c(1:end-2,2:end-1)+f_c(3:end,2:end-1)+...
%             f_c(2:end-1,1:end-2)+f_c(2:end-1,3:end);
%         d_f_c=rho*(lambda3*(extf_c.*(1 + jac) + i_m_1y + j_m_1y + i_p_1y + j_p_1y) + lambda2*barrierx + lambda*intf_c);
%         d_f_c=min(max(d_f_c,-3),3); %don't want to jump too much in one iteration
%         f_c_temp(3:end-2,3:end-2)=f_c(3:end-2,3:end-2)+d_f_c(3:end-2,3:end-2);
%         
%         f_c_temp=max(min(f_c_temp,imgsize(1)),1); %edge control
%         
%         f_r_temp = f_r;
%         intf_r(2:end-1,2:end-1)= -4*f_r(2:end-1,2:end-1)+ ...
%             f_r(1:end-2,2:end-1)+f_r(3:end,2:end-1)+...
%             f_r(2:end-1,1:end-2)+f_r(2:end-1,3:end);
%         d_f_r=rho*(lambda3*(extf_r.*(1 + jac) + i_m_1x + j_m_1x + i_p_1x + j_p_1x) + lambda2*barriery +lambda*intf_r);
%         d_f_r=min(max(d_f_r,-3),3);
%         f_r_temp(3:end-2,3:end-2)=f_r(3:end-2,3:end-2)+d_f_r(3:end-2,3:end-2);
%         
%         f_r_temp=max(min(f_r_temp,imgsize(1)),1);
%         
%         f_c_temp = gather(f_c_temp);
%         f_r_temp = gather(f_r_temp);
%         
%         save('out.mat');
%         
%         jac = ones(size(f_c));
%         jac(2:end-1,2:end-1) = .25 * ((f_c_temp(2:end-1,3:end) - f_c_temp(2:end-1,1:end-2)).*...
%             (f_r_temp(3:end,2:end-1) - f_r_temp(1:end-2,2:end-1))-...
%             ((f_c_temp(3:end,2:end-1) - f_c_temp(1:end-2,2:end-1)).*...
%             (f_r_temp(2:end-1,3:end) - f_r_temp(2:end-1,1:end-2))));
            
        if (min(jacf_temp(:)) >= 0 && min(jacg_temp(:)) >= 0)
            f_c = f_c_temp;
            f_r = f_r_temp;
            g_r = g_r_temp;
            g_c = g_c_temp;
            jacf = jacf_temp;
            jacg = jacg_temp;
            break;
        else
            rho = rho * .75;
        end
    end
    
    
    
    
    
end

f_c = gather(f_c);
g_c = gather(g_c);
jacf = gather(jacf);
jacg = gather(jacg);
fbarrierx = gather(fbarrierx);
gbarrierx = gather(gbarrierx);
img2_o_f = gather(img2_o_f);
img1_o_g = gather(img1_o_g);
extf_c = gather(extf_c);
extg_c = gather(extg_c);
disp(rho);

save('out/out.mat');

img1 = gather(img1);


% 
% % figure('visible','off');
% % subplot(1,2,1);
% % imshow(mat2gray(img2_o_f));
% % subplot(1,2,2);
% % tmp=-img1+img2_o_f;
% % tmp(1,1)=1;
% % imshow(mat2gray(tmp));
% % print -dpng 'diff.png'
% 
imwrite(mat2gray(img2_o_f), 'out/interpolated2.png', 'png');
imwrite(mat2gray(img1_o_g), 'out/interpolated1.png', 'png');
imwrite(mat2gray(img2_o_f - img1_o_g), 'out/diff.png', 'png');


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
% 
% function jac = jacobian(f_c,f_r)
% jac = ones(size(f_c));
% jac(2:end-1,2:end-1) = .25 * ((f_c(2:end-1,3:end) - f_c(2:end-1,1:end-2)).*...
%     (f_r(3:end,2:end-1) - f_r(1:end-2,2:end-1))-...
%     ((f_c(3:end,2:end-1) - f_c(1:end-2,2:end-1)).*...
%     (f_r(2:end-1,3:end) - f_r(2:end-1,1:end-2))));

