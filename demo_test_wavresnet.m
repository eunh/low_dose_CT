%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Made by Eunhee Kang (eunheekang@kaist.ac.kr) at 2017.12.08
% 2017 Fully3D Paper: 'Wavelet Domain Residual Network (WavResNet) for Low-Dose X-ray CT Reconstruction'
% Author: Eunhee Kang, Junhong Min, and Jong Chul Ye
% Bio Imaging and Signal Processing Lab., Dept. of Bio and Brain Engineering, KAIST
% 
% Copyright <2017> <Eunhee Kang (eunheekang@kaist.ac.kr)>
% 
% Redistribution and use in source and binary forms, with or without modification, are permitted provided that the following conditions are met:
% 
% 1. Redistributions of source code must retain the above copyright notice, this list of conditions and the following disclaimer.
% 
% 2. Redistributions in binary form must reproduce the above copyright notice, this list of conditions and the following disclaimer in the documentation and/or other materials provided with the distribution.
% 
% 3. Neither the name of the copyright holder nor the names of its contributors may be used to endorse or promote products derived from this software without specific prior written permission.
% 
% THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" 
% AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, 
% THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. 
% IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, 
% OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; 
% LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, 
% WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, 
% EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
% 
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%%
clear;
close all;

%% Path setting
addpath(genpath('lib_contourlet')); % Contourlet transform library path
run('matconvnet-1.0-beta24\matlab\vl_setupnn.m'); % MatConvNet path

%% Load network and test data
% network
load('trained_networks\net-forward-process.mat');

% test data 
nTestCase = 1; % 1: lung, 2: liver, 3: pelvic bone
load(['test_data\test_case' num2str(nTestCase) '.mat']);

%% Parameters
lv                  = [1,2,3];              % vector of numbers of directional filter bank decomposition levels at each pyramidal level
dflt                = 'vk';                 % filter name for the directional decomposition step
patchsize           = 55;                   % the size of patch
batchsize           = 10;                   % the size of batch
overlap             = 10;                   % the size of overlap region to recon the whole image(512x512)
wgt                 = 1e3;                  % weight multiplied to input
gpus                = 1;                    % gpu on / off

%% Test
% GPU reset
if gpus > 0
    reset(gpuDevice(gpus));
    net = vl_simplenn_move(net, 'gpu');
end

recon = cnn_CT_denoising_forward_process(net,quarter_dose,lv,dflt,patchsize,batchsize,overlap,wgt,gpus);

%% Plot
% intensity convert to Hounsfield Unit
hu_quarter  = (quarter_dose - 0.0192)/0.0192*1000;
hu_recon    = (recon - 0.0192)/0.0192*1000;
hu_routine  = (routine_dose - 0.0192)/0.0192*1000;

% image metric
psnr_input = psnr(quarter_dose, routine_dose, max(routine_dose(:)));
ssim_input = ssim(quarter_dose, routine_dose, 'DynamicRange', max(routine_dose(:)));
psnr_recon = psnr(recon, routine_dose, max(routine_dose(:)));
ssim_recon = ssim(recon, routine_dose, 'DynamicRange', max(routine_dose(:)));

wndVal = [-160 240];
figure(1);      colormap gray;
subplot(131);   imagesc(hu_quarter,wndVal);     axis image off;     title({'Input: Quarter-dose';...
                                                                           ['PNSR [dB]: ' num2str(psnr_input)];...
                                                                           ['SSIM index: ' num2str(ssim_input)]});
subplot(132);   imagesc(hu_recon,wndVal);       axis image off;     title({'Recon';...
                                                                           ['PNSR [dB]: ' num2str(psnr_recon)];...
                                                                           ['SSIM index: ' num2str(ssim_recon)]});
subplot(133);   imagesc(hu_routine,wndVal);     axis image off;     title('Label: Routine-dose');