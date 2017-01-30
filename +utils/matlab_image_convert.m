function caffe_image = matlab_image_convert(matlab_image, image_size, image_mean)
% Convert an image returned by Matlab's imread to caffe_image in caffe's data
% Output format: W x H x C with BGR channels

% permute channels from RGB to BGR
caffe_image = matlab_image(:, :, [3, 2, 1]);
% resize im_data
caffe_image = imresize(caffe_image, image_size, 'bilinear');
% flip width and height
caffe_image = permute(caffe_image, [2, 1, 3]);  
% convert from uint8 to single
caffe_image = single(caffe_image);
% subtract mean_data (BGR)
if (length(image_mean(:)) == 3)
    % if only rgb mean, repmat the mean vector
    caffe_image = caffe_image - repmat(reshape(image_mean, [1, 1, 3]), [size(caffe_image, 1), size(caffe_image, 2), 1]);
else
    caffe_image = caffe_image - image_mean;
end