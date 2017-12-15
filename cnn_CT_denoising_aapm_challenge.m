function recon = cnn_CT_denoising_aapm_challenge(net,noisy,lv,dflt,patchsize,batchsize,overlap,wgt,gpus)


ind         = 1 : size(noisy,3);
noisyCoeffs = single(cnn_wavelet_decon(double(noisy),lv,dflt));

noisyCoeffs = noisyCoeffs .*wgt;

[ny, nx]    = size(noisy);
wgtMap      = zeros(ny, nx, 'single');
reconCoeffs = zeros(size(noisyCoeffs),'single');

if gpus > 0
    reconCoeffs = gpuArray(reconCoeffs);
end

for y = 1:(patchsize-overlap):ny-1
    yy = min(y, ny-patchsize+1);
    
    for x = 1:(patchsize-overlap):nx-1
        xx = min(x,nx-patchsize+1);
        
        wgtMap(yy:(yy+patchsize-1),xx:(xx+patchsize-1),:) = ...
            wgtMap(yy:(yy+patchsize-1),xx:(xx+patchsize-1),:) +1;
        
        for tt = 1:ceil(length(ind)/batchsize)
            ind_s = batchsize*(tt-1)+1;
            ind_e = min(length(ind),ind_s+batchsize-1);
            
            noisyCoeffsSub = noisyCoeffs(yy:(yy+patchsize-1),xx:(xx+patchsize-1),:,ind_s:ind_e);
            if gpus > 0
                noisyCoeffsSub = gpuArray(noisyCoeffsSub);
                res = vl_simplenn_modified(net,single(noisyCoeffsSub),[],[],...
                'mode','test',...
                'conserveMemory', 1, ...
                'cudnn', 1);
            else
                res = vl_simplenn_modified(net,single(noisyCoeffsSub),[],[],...
                'mode','test',...
                'conserveMemory', 1);
            end
            
            reconCoeffsSub = res(end-1).x;
            
            reconCoeffs(yy:(yy+patchsize-1),xx:(xx+patchsize-1),:,ind_s:ind_e) =...
                reconCoeffs(yy:(yy+patchsize-1),xx:(xx+patchsize-1),:,ind_s:ind_e) + reconCoeffsSub;
        end
        
    end
    
end

wgtMap = repmat(wgtMap, [1, 1, size(reconCoeffs,3)]);
if gpus > 0
    reconCoeffs = gather(reconCoeffs);
end

for tt = 1: size(reconCoeffs,4)
    reconCoeffs(:,:,:,tt) = reconCoeffs(:,:,:,tt)./wgtMap;
end

reconCoeffs(:,:,1,:) = reconCoeffs(:,:,1,:) + noisyCoeffs(:,:,1,:);

recon  = single(cnn_wavelet_recon(double(reconCoeffs./wgt),lv,dflt));
recon(recon < 0) = 0;

