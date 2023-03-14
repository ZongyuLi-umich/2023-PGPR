
img_name = '12';
img = imread(sprintf('/Users/xuxiaojian/Downloads/set12/%s.png', img_name));
img = imresize(img,[256 256]);
imwrite(img, sprintf('/Users/xuxiaojian/Downloads/set11/%s.png', img_name), 'png')
