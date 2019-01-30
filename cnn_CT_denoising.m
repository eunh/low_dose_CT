% -------------------------------------------------------------------------
function [net, info] = cnn_CT_denoising(imdb, varargin)
% -------------------------------------------------------------------------

opts.method             = 'image';

[opts, varargin]        = vl_argparse(opts, varargin);

opts.expDir             = '';
opts.train              = struct();
[opts, varargin]        = vl_argparse(opts, varargin);

if ~isfield(opts.train, 'gpus'), opts.train.gpus = []; end;

opts.wgt                = 1;

opts.numEpochs          = 1e2;
opts.num_lr_epoch       = 1e3;
opts.patchSize          = 50;
opts.batchSize          = 16;
opts.numSubBatches      = 1;
opts.batchSample        = 1;
opts.numTrainDB         = 200;
opts.iterations         = [];

opts.lrnrate            = [-3, -5];
opts.wgtdecay           = 1e-4;
opts.gradMax            = 1e-2;

opts.solver             = [];

opts.weightInitMethod       = 'gaussian';
opts.networkType            = 'simplenn';
opts.batchNormalization     = true ;

[opts, ~]               = vl_argparse(opts, varargin);


net = cnn_CT_denoising_contourlet_init( ...
    'wgt',          opts.wgt,        'method',       opts.method, ...
    'patchSize',    opts.patchSize,  'batchSize',    opts.batchSize,  'numSubBatches',     opts.numSubBatches,   ...
    'lrnrate',      opts.lrnrate,    'weightDecay',  opts.wgtdecay,   'solver',            opts.solver,          ...
    'gradMax',      opts.gradMax,    'numTrainDB',   opts.numTrainDB, 'weightInitMethod', opts.weightInitMethod, ...
    'iterations',   opts.iterations, 'numEpochs',    opts.numEpochs);

switch opts.networkType
    case 'simplenn', trainFn = @cnn_train_modified;
    case 'dagnn', trainFn = @cnn_train_dag;
end

[net, info] = trainFn(net, imdb, getBatch(opts), ...
    'expDir', opts.expDir, ...
    net.meta.trainOpts, ...
    opts.train, ...
    'val', find(imdb.images.set==2));

% -------------------------------------------------------------------------
function fn = getBatch(opts)
% -------------------------------------------------------------------------
bopts = struct('numGpus', numel(opts.train.gpus), 'method', opts.method, 'wgt', opts.wgt, 'expDir', opts.expDir, ...
    'patchSize', opts.patchSize, 'batchSize', opts.batchSize);
fn = @(x,y) getSimpleNNBatch(bopts,x,y) ;

% -------------------------------------------------------------------------
function [images, labels] = getSimpleNNBatch(opts, imdb, batch)
% -------------------------------------------------------------------------
method      = opts.method; % to determine the label type ('image', 'residual')
wgt         = opts.wgt;
patchSize   = opts.patchSize;

Ny          = size(imdb.images.data,1);
Nx          = size(imdb.images.data,2);
pos_x       = round(rand(1)*(Nx-patchSize));
pos_y       = round(rand(1)*(Ny-patchSize-80))+30;

images = single(imdb.images.data(pos_y+(1:patchSize),pos_x+(1:patchSize),:,batch));
labels = single(imdb.images.label(pos_y+(1:patchSize),pos_x+(1:patchSize),:,batch));

labels = labels.*wgt;
images = images.*wgt;

if strcmp(method, 'residual') 
    labels  = labels - images;
end

% augmentation - flip(data, dim)
if (rand > 0.5)
    images  = flip(images, 1);
    labels  = flip(labels, 1);
end

if (rand > 0.5)
    images 	= flip(images, 2);
    labels  = flip(labels, 2);
end

if opts.numGpus > 0
    images = gpuArray(images);
    labels = gpuArray(labels);
end