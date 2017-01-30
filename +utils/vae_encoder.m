function zt = vae_encoder(xt, caffe_vae_weights)
% Get zt through MLP encoder
% Input format
%       xt: (50, 512)
% Output format
%       zt: (50, h_dim)

top = xt;
for layer_id = 1 : length(caffe_vae_weights.encoder)
    % caffe weight (1, 1, C, N)
    weight = caffe_vae_weights.encoder(layer_id).w;
    % reshape weight (C, N)
    weight = reshape(weight, [size(weight, 3), size(weight, 4)]);
    % caffe bias (N, 1)
    bias = caffe_vae_weights.encoder(layer_id).b;
    top = top * weight + repmat(bias', [size(top, 1), 1]); 
    if (strcmp(caffe_vae_weights.encoder(layer_id).act, 'ReLU'))
        top = top .* (top > 0);
    end
end
zt = top;