%% set fv_vae parameters
% extractor network prototxt
fv_vae_opt.network_deploy_prototxt = './pretrained_models/vgg19_deploy.prototxt';
% extractor network weights
fv_vae_opt.network_weights = './pretrained_models/VGG_ILSVRC_19_layers.caffemodel';
% caffe vae weights
fv_vae_opt.caffe_vae_weights = './vae_models/caffe_vae_weights_ucf101_split1_vgg19.mat';
% caffe matlab path
fv_vae_opt.caffe_matlab_path = './caffe/matlab';
% caffe running device id (<0 => run on cpu)
fv_vae_opt.device = 0;

%% initial extractor network
addpath(fv_vae_opt.caffe_matlab_path);
if (fv_vae_opt.device < 0)
    caffe.set_mode_cpu();
else
    caffe.set_mode_gpu();
    caffe.set_device(fv_vae_opt.device);
end
caffe.reset_all();
extractor_net = caffe.Net(fv_vae_opt.network_deploy_prototxt, 'test');
extractor_net.copy_from(fv_vae_opt.network_weights);

%% input image path
frame_path = {'./example_images/000001.jpg'; './example_images/000002.jpg';
    './example_images/000003.jpg';  './example_images/000004.jpg';
    './example_images/000005.jpg';  './example_images/000006.jpg';
    './example_images/000007.jpg';  './example_images/000008.jpg';
    './example_images/000009.jpg';  './example_images/000010.jpg'};

% set image convert parameters
resize_h = 240;
resize_w = 320;
crop_size = 224;
% load vae parameters from caffe
load(fv_vae_opt.caffe_vae_weights);
x_dim = 512;
h_dim = length(caffe_vae_weights.encoder(end).b);
sig = caffe_vae_weights.decoder_sigma;

%% average pooling raw_fv of all frames
for frame_id = 1 : length(frame_path)
    % convert matlat format image to caffe format image (W, H, C)
    caffe_image = utils.matlab_image_convert(imread(frame_path{frame_id}), [resize_h, resize_w], [104, 117, 123]);
    % center crop caffe_image (CROP, CROP, C, 1)
    begin_h = floor((resize_h - crop_size) / 2);
    begin_w = floor((resize_w - crop_size) / 2);
    input_blob = reshape(caffe_image(begin_w + 1 : begin_w + crop_size, begin_h + 1 : begin_h + crop_size, :), ...
        [crop_size, crop_size, 3, 1]);
    
    % feed input_blob to extractor network
    input_blobs = { input_blob };
    output_blobs = extractor_net.forward(input_blobs);
    % get output_blob (7, 7, 512)
    output_blob = output_blobs{1};
    
    % pre-normalization (7, 7, 512)
    activations = output_blob / sqrt(output_blob(:)' * output_blob(:)) * caffe_vae_weights.norm_scale;
    % spp (xt) (50, 512)
    xt = utils.activations_spp(activations);
    % get zt through MLP encoder (50, h_dim)
    zt = utils.vae_encoder(xt, caffe_vae_weights);
    % get uxt through decoder (50, 512)
    uxt = utils.vae_decoder(zt, caffe_vae_weights);
    % calcuate un-normalized FV raw_fv (1, 512 * (h_dim + 1)) in Eq.11
    fv_mat =  ((uxt - xt) ./ repmat(sig' .* sig', [50, 1]) .* (uxt > 0))' * [zt, ones(50, 1)];
    if (frame_id == 1)
        raw_fv = fv_mat(:) / length(frame_path);
    else
        raw_fv = raw_fv + fv_mat(:) / length(frame_path);
    end
end
