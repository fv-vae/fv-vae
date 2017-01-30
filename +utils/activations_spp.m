function spp_output = activations_spp(activations)
% Do spatial pyramid pooling on activations
% Input format
%       activations: (7, 7, C)
% Output format
%       spp_output: (50, C)

C = size(activations, 3);

% p1 2x2 (stride 1x1) pooling (36, C)
p1 = zeros(6, 6, C);
for w = 1 : 6
    for h = 1 : 6
        tw = w;
        th = h;
        p1(w, h, :) = max(max(activations(tw : tw + 1, th : th + 1, :), [], 1), [], 2);
    end
end
p1 = reshape(p1, [36, C]);

% p2 3x3 (stride 2x2) pooling (9, C)
p2 = zeros(3, 3, C);
for w = 1 : 3
    for h = 1 : 3
        tw = w * 2 - 1;
        th = h * 2 - 1;
        p2(w, h, :) = max(max(activations(tw : tw + 2, th : th + 2, :), [], 1), [], 2);
    end
end
p2 = reshape(p2, [9, C]);

% p3 4x4 (stride 3x3) pooling (4, C)
p3 = zeros(2, 2, C);
for w = 1 : 2
    for h = 1 : 2
        tw = w * 3 - 2;
        th = h * 3 - 2;
        p3(w, h, :) = max(max(activations(tw : tw + 3, th : th + 3, :), [], 1), [], 2);
    end
end
p3 = reshape(p3, [4, C]);

% p4 7x7 global pooling (1, C)
p4 = max(max(activations, [], 1), [], 2);
p4 = reshape(p4, [1, C]);

spp_output = [p1; p2; p3; p4];