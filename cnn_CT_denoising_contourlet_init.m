function net = cnn_CT_denoising_contourlet_init(varargin)
% cnn_CT_denoising_contourlet_init  Initialize a WavResNet for Low-dose CT

opts.method = 'image';

[opts, ~]   = vl_argparse(opts, varargin);

opts.wgt                    = 1;

opts.scale                  = 1 ;
opts.weightDecay            = 1e-2 ;
opts.weightInitMethod       = 'gaussian' ;
opts.networkType            = 'simplenn' ;
opts.cudnnWorkspaceLimit    = 1024*1024*1204*10 ; % 1GB

opts.numEpochs          = 1e2;
opts.num_lr_epoch       = 1e3;
opts.patchSize          = 50;
opts.batchSize          = 16;
opts.numSubBatches      = 1;
opts.numTrainDB         = 200;
opts.iterations         = [];

opts.lrnrate            = [-3, -5];
opts.gradMax            = 1e-2;

opts.solver             = [];

[opts, ~]               = vl_argparse(opts, varargin) ;
%%
net = [];
net = wavRes(net, opts) ;

% final touches
switch lower(opts.weightInitMethod)
    case {'xavier', 'xavierimproved'}
        net.layers{end}.weights{1} = net.layers{end}.weights{1} / 10 ;
end
% loss
net.layers{end+1} = struct('type', 'euclideanloss', 'name', 'loss') ;

%%
% Meta parameters
net.meta.trainOpts.learningRate     = logspace(opts.lrnrate(1), opts.lrnrate(2), opts.num_lr_epoch) ;
net.meta.trainOpts.batchSize        = opts.batchSize ;
net.meta.trainOpts.numSubBatches    = opts.numSubBatches;
net.meta.trainOpts.patchSize        = opts.patchSize ;
net.meta.trainOpts.numEpochs        = opts.numEpochs ;
net.meta.trainOpts.wgt              = opts.wgt ;
net.meta.trainOpts.weightDecay      = opts.weightDecay ;
net.meta.trainOpts.gradMax          = opts.gradMax ;
net.meta.trainOpts.solver           = opts.solver ;
net.meta.trainOpts.momentum         = 9e-1;
net.meta.trainOpts.method           = opts.method ;
net.meta.trainOpts.numTrainDB       = opts.numTrainDB ;
net.meta.trainOpts.iterations       = opts.iterations ;

% Fill in default values
net = vl_simplenn_tidy(net) ;

% Switch to DagNN if requested
switch lower(opts.networkType)
    case 'simplenn'
        % done
    case 'dagnn'
        net = dagnn.DagNN.fromSimpleNN(net, 'canonicalNames', true) ;
        net.addLayer('top1err', dagnn.Loss('loss', 'classerror'), ...
            {'prediction','label'}, 'top1err') ;
        net.addLayer('top5err', dagnn.Loss('loss', 'topkerror', ...
            'opts', {'topK',5}), ...
            {'prediction','label'}, 'top5err') ;
    otherwise
        assert(false) ;
end
end

function net = add_conv(net, opts, id, h, w, in, out, stride, pad)
% --------------------------------------------------------------------
info = vl_simplenn_display(net) ;
fc = (h == info.dataSize(1,end) && w == info.dataSize(2,end)) ;
if fc
    name = 'fc' ;
else
    name = 'conv' ;
end
convOpts = {'CudnnWorkspaceLimit', opts.cudnnWorkspaceLimit} ;
net.layers{end+1} = struct('type', 'conv', 'name', sprintf('%s%s', name, id), ...
    'weights', {{init_weight(opts, h, w, in, out, 'single'), zeros(out, 1, 'single')}}, ...
    'stride', stride, ...
    'pad', pad, ...
    'learningRate', [1 2], ...
    'weightDecay', [opts.weightDecay 0], ...
    'opts', {convOpts}) ;
end

% --------------------------------------------------------------------
function net = add_block(net, opts, id, h, w, in, out, stride, pad,batchOn,ReluOn)
% --------------------------------------------------------------------
info = vl_simplenn_display(net) ;
fc = (h == info.dataSize(1,end) && w == info.dataSize(2,end)) ;
if fc
    name = 'fc' ;
else
    name = 'conv' ;
end
convOpts = {'CudnnWorkspaceLimit', opts.cudnnWorkspaceLimit} ;
net.layers{end+1} = struct('type', 'conv', 'name', sprintf('%s%s', name, id), ...
    'weights', {{init_weight(opts, h, w, in, out, 'single'), zeros(out, 1, 'single')}}, ...
    'stride', stride, ...
    'pad', pad, ...
    'dilate', 1, ...
    'learningRate', [1 2], ...
    'weightDecay', [opts.weightDecay 0], ...
    'opts', {convOpts}) ;
if batchOn
    net.layers{end+1} = struct('type', 'bnorm', 'name', sprintf('bn%s',id), ...
        'weights', {{ones(out, 1, 'single'), zeros(out, 1, 'single'), zeros(out, 2, 'single')}}, ...
        'learningRate', [2 1 0.05], ...
        'weightDecay', [0 0]) ;
end
if ReluOn
    net.layers{end+1} = struct('type', 'relu', 'name', sprintf('relu%s',id)) ;
end
end

% -------------------------------------------------------------------------
function weights = init_weight(opts, h, w, in, out, type)
% -------------------------------------------------------------------------
% See K. He, X. Zhang, S. Ren, and J. Sun. Delving deep into
% rectifiers: Surpassing human-level performance on imagenet
% classification. CoRR, (arXiv:1502.01852v1), 2015.

switch lower(opts.weightInitMethod)
    case 'gaussian'
        sc = 0.001/opts.scale ;
        weights = randn(h, w, in, out, type)*sc;
    case 'xavier'
        sc = sqrt(3/(h*w*in)) ;
        weights = (rand(h, w, in, out, type)*2 - 1)*sc ;
    case 'xavierimproved'
        sc = sqrt(2/(h*w*out)) ;
        weights = randn(h, w, in, out, type)*sc ;
    otherwise
        error('Unknown weight initialization method''%s''', opts.weightInitMethod) ;
end
end

% --------------------------------------------------------------------
function net = add_norm(net, opts, id)
% --------------------------------------------------------------------

net.layers{end+1} = struct('type', 'normalize', ...
    'name', sprintf('norm%s', id), ...
    'param', [5 1 0.0001/5 0.75]) ;
end

% --------------------------------------------------------------------
function net = add_dropout(net, opts, id)
% --------------------------------------------------------------------
net.layers{end+1} = struct('type', 'dropout', ...
    'name', sprintf('dropout%s', id), ...
    'rate', 0.5) ;
end

function net = add_reg_catch(net, id,regNum,reluOn)
% --------------------------------------------------------------------
net.layers{end+1} = struct('type', 'reg_catch', ...
    'name', sprintf('reg%s', id),'regNum',regNum) ;
if reluOn
    net.layers{end+1} = struct('type', 'relu', 'name', sprintf('relu%s',id)) ;
end
end

function net = add_reg_toss(net, id, regNum)
% --------------------------------------------------------------------
net.layers{end+1} = struct('type', 'reg_toss', ...
    'name', sprintf('reg%s', id),...
    'regNum', regNum) ;
end

function net = add_reg_concat(net, id, regSet)
% --------------------------------------------------------------------
net.layers{end+1} = struct('type', 'reg_concat', ...
    'name', sprintf('reg%s', id),...
    'regSet', regSet) ;

end

% --------------------------------------------------------------------
function net = wavRes(net, opts)
% --------------------------------------------------------------------

net.layers = {} ;
ch_inp      = 15;
ch_pro      = 128;
KerenlSizeY = 3;
KerenlSizeX = 3;
zeroPadY = floor(KerenlSizeY/2);
zeroPadX = floor(KerenlSizeX/2);
zeroPad  = [zeroPadY zeroPadY zeroPadX zeroPadX];

net = add_block(net, opts, '0_1', KerenlSizeY, KerenlSizeX, ch_inp, ch_pro, 1, zeroPad,1,1) ;
net = add_block(net, opts, '0_2', KerenlSizeY, KerenlSizeX, ch_pro, ch_pro, 1, zeroPad,1,1) ;
net = add_block(net, opts, '0_3', KerenlSizeY, KerenlSizeX, ch_pro, ch_pro, 1, zeroPad,1,1) ;

net = add_reg_toss(net, '1',1);
net = add_block(net, opts, '1_1', KerenlSizeY, KerenlSizeX, ch_pro, ch_pro, 1, zeroPad,1,1) ;
net = add_block(net, opts, '1_2', KerenlSizeY, KerenlSizeX, ch_pro, ch_pro, 1, zeroPad,1,1) ;
net = add_block(net, opts, '1_3', KerenlSizeY, KerenlSizeX, ch_pro, ch_pro, 1, zeroPad,1,1) ;
net = add_reg_catch(net, '1',1, 1);

net = add_reg_toss(net, '2',2);
net = add_block(net, opts, '2_1', KerenlSizeY, KerenlSizeX, ch_pro, ch_pro, 1, zeroPad,1,1) ;
net = add_block(net, opts, '2_2', KerenlSizeY, KerenlSizeX, ch_pro, ch_pro, 1, zeroPad,1,1) ;
net = add_block(net, opts, '2_3', KerenlSizeY, KerenlSizeX, ch_pro, ch_pro, 1, zeroPad,1,1) ;
net = add_reg_catch(net, '2',2,1);

net = add_reg_toss(net, '3',3);
net = add_block(net, opts, '3_1', KerenlSizeY, KerenlSizeX, ch_pro, ch_pro, 1, zeroPad,1,1) ;
net = add_block(net, opts, '3_2', KerenlSizeY, KerenlSizeX, ch_pro, ch_pro, 1, zeroPad,1,1) ;
net = add_block(net, opts, '3_3', KerenlSizeY, KerenlSizeX, ch_pro, ch_pro, 1, zeroPad,1,1) ;
net = add_reg_catch(net, '3',3,1);

net = add_reg_toss(net, '4',4);
net = add_block(net, opts, '4_1', KerenlSizeY, KerenlSizeX, ch_pro, ch_pro, 1, zeroPad,1,1) ;
net = add_block(net, opts, '4_2', KerenlSizeY, KerenlSizeX, ch_pro, ch_pro, 1, zeroPad,1,1) ;
net = add_block(net, opts, '4_3', KerenlSizeY, KerenlSizeX, ch_pro, ch_pro, 1, zeroPad,1,1) ;
net = add_reg_catch(net, '4',4,1);

net = add_reg_toss(net, '5',5);
net = add_block(net, opts, '5_1', KerenlSizeY, KerenlSizeX, ch_pro, ch_pro, 1, zeroPad,1,1) ;
net = add_block(net, opts, '5_2', KerenlSizeY, KerenlSizeX, ch_pro, ch_pro, 1, zeroPad,1,1) ;
net = add_block(net, opts, '5_3', KerenlSizeY, KerenlSizeX, ch_pro, ch_pro, 1, zeroPad,1,1) ;
net = add_reg_catch(net, '5',5,1);

net = add_reg_toss(net, '6',6);
net = add_block(net, opts, '6_1', KerenlSizeY, KerenlSizeX, ch_pro, ch_pro, 1, zeroPad,1,1) ;
net = add_block(net, opts, '6_2', KerenlSizeY, KerenlSizeX, ch_pro, ch_pro, 1, zeroPad,1,1) ;
net = add_block(net, opts, '6_3', KerenlSizeY, KerenlSizeX, ch_pro, ch_pro, 1, zeroPad,1,1) ;
net = add_reg_catch(net, '6',6,1);

net = add_reg_concat(net, '7_0',1:6);
net = add_block(net, opts, '7_1', KerenlSizeY, KerenlSizeX, ch_pro*7, ch_pro, 1, zeroPad,1,1) ;
net = add_block(net, opts, '7_2', KerenlSizeY, KerenlSizeX, ch_pro, ch_pro, 1, zeroPad,1,1) ;
net = add_block(net, opts, '7_3', KerenlSizeY, KerenlSizeX, ch_pro, ch_inp, 1, zeroPad,0,0) ;

info = vl_simplenn_display(net);
net.meta.regNum = 6;
net.meta.regSize = [info.dataSize(1,end),info.dataSize(2,end),info.dataSize(3,end)];

end