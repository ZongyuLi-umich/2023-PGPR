I = imread('/Users/xuxiaojian/Downloads/virus.tif');
rect_all = [8.5100  112.5100  640.0000  640.0000;...
            719.5100  112.5100  640.0000  640.0000;...
            1439.5    113.5    640.0000  640.0000;...
            8.5100  989.5100  640.0000  640.0000;...
            725.5    1085.5    640.0000  640.0000;...
            1439.5    989.5    640.0000  640.0000;...
            14.5    1884.5    640.0000  640.0000;...
            725.5    1872.5    640.0000  640.0000;...
            1451.5    1892.5    640.0000  640.0000];

idx = 1;
%% Method 1: use software to crop
[J,rect] = imcrop(I);
rect(3) = 640;
rect(4) = 640;
J = imcrop(I,rect);

figure()
subplot(1,2,1)
imshow(I)
title('Original Image')
subplot(1,2,2)
imshow(J)
title('Cropped Image')

%% use pre-defined rect to crop
J = imcrop(I,rect_all(idx, :));
figure()
subplot(1,2,1)
imshow(I)
title('Original Image')
subplot(1,2,2)
imshow(J)
title('Cropped Image')

%% save image
imwrite(J,sprintf('/Users/xuxiaojian/Downloads/img_%d.tiff', idx), 'tiff');

