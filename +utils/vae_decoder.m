function uxt = vae_decoder(zt, caffe_vae_weights)
% Get uxt through decoder
% Input format
%       xt: (50, 512)
% Output format
%       zt: (50, h_dim)

% caffe weight (1, 1, C, N)
weight = caffe_vae_weights.decoder.w;
% reshape weight (C, N)
weight = reshape(weight, [size(weight, 3), size(weight, 4)]);
% caffe bias (N, 1)
bias = caffe_vae_weights.decoder.b;
uxt = zt * weight' + repmat(bias', [size(zt, 1), 1]); 
if (strcmp(caffe_vae_weights.decoder.act, 'ReLU'))
    uxt = uxt .* (uxt > 0);
end