
%%
clear;
close all;

%% Path setting
addpath(genpath('lib_contourlet')); % Contourlet transform library path
run('matconvnet-1.0-beta24\matlab\vl_setupnn.m'); % MatConvNet path

%% Parameters
lv                  = [1,2,3];              % vector of numbers of directional filter bank decomposition levels at each pyramidal level
dflt                = 'vk';                 % filter name for the directional decomposition step
patchsize           = 55;                   % the size of patch
batchsize           = 10;                   % the size of batch
wgt                 = 1e3;                  % weight multiplied to input
num_epoch           = 500;                  % the number of epochs
lr_rate             = [-2 -5];              % learing rate scheduling from 1e-2 to 1e-5
num_lr_epoch        = num_epoch;            
wgtdecay            = 1e-2;                 % weight decay parameter
gradMax             = 1e-3;                 % gradient clipping

gpus                = 1;                    % gpu on / off
train               = struct('gpus', gpus);

expdir              = 'training_1';         % experiment name
if ~exist(expdir, 'dir'), mkdir(expdir); end;

%% Train

% prepare the dataset (imdb) 
% imdb.images.data: input data (contourlet transform coefficients of low-dose CT images, size=[nx ny nch nbatch]) 
% imdb.images.label: target data (contourlet transform coefficients of routine-dose CT images) 
% imdb.images.set: 1D vector for data (1: training data, 2: validation data, 3: test data)
% imdb.meta.sets: {'train' 'val' 'test'}

[net, info] = cnn_CT_denoising(imdb, ...
    'expDir',       expdir,     'method',       'residual',             ...
    'numEpochs',    num_epoch,     ...
    'patchSize',    patchsize,  'batchSize',    batchsize,              ...
    'wgt',          wgt,        'lrnrate',      lr_rate,                ...
    'wgtdecay',     wgtdecay,   ...
    'gradMax',      gradMax,    'num_lr_epoch', num_lr_epoch,           ...
    'train',        train);
