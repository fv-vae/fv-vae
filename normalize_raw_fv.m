function fv_vae = normalize_raw_fv(raw_fv)
% Normalize raw_fv
% Input format
%       raw_fv: (N, d) 
%                 N is the number of samples in the dataset
% Output format
%       fv_vae: (N, d)

[N, d] = size(raw_fv);
% small float to avoid "divided by 0"
eps = 1e-10;
% do column l2 normalization to approximate FIM normalization
fv_vae = raw_fv ./ repmat(max(sqrt(sum(raw_fv .* raw_fv, 1)), eps), [N, 1]);
% do signed square-root step
fv_vae = sign(fv_vae) .* sqrt(abs(fv_vae));
% do row l2 normalization
fv_vae = fv_vae ./ repmat(max(sqrt(sum(fv_vae .* fv_vae, 2)), eps), [1, d]);
