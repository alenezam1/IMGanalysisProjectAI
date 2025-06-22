%% Image Processing, Classification, and Clustering Demo
% End-to-end exploration of edge detection, K-Nearest Neighbour (KNN)
% classification, and super-pixel segmentation in MATLAB.

clear all; close all; clc;

%% -----------------------------------------------------------
%% 1. Edge Detection & Laplacian-of-Gaussian Filtering
%% -----------------------------------------------------------

% Load and prepare grayscale logo
logo   = mean(double(imread('LancsLogo.jpg'))/255, 3);
logo   = logo(276:end-250, :);              % crop lower banner
figure(1), colormap gray
subplot(3,1,1), imagesc(logo), axis image off title('Input Logo');

% Simple high-pass kernel
hpKer = [-1 -1 -1; -1 8 -1; -1 -1 -1];
hpImg = conv2(logo, hpKer, 'same');
subplot(3,1,2), imagesc(abs(hpImg)), axis image off title('High-Pass Response');

% Threshold edge map
th     = 0.15;
edges  = abs(hpImg) >= th;
subplot(3,1,3), imagesc(edges), axis image off title('Binary Edge Map');

% Laplacian-of-Gaussian kernel
grid  = linspace(-4,4,9);
[X,Y] = meshgrid(grid,grid);
sig   = 1.4;
LoG   = -1/(pi*sig^4) .* (1-(X.^2+Y.^2)/(2*sig^2)) .* exp(-(X.^2+Y.^2)/(2*sig^2));

% Landscape demo
land  = mean(double(imread('Landscape.jpg'))/255,3);
landF = conv2(land, LoG, 'same');
figure(2), colormap gray
subplot(2,1,1), imagesc(land),  axis image off title('Landscape');
subplot(2,1,2), imagesc(abs(landF)), axis image off title('LoG Response');

% Building demo
build  = mean(double(imread('InfoLab.jpg'))/255,3);
buildF = conv2(build, LoG, 'same');
figure(3), colormap gray
subplot(2,1,1), imagesc(build),  axis image off title('Building');
subplot(2,1,2), imagesc(abs(buildF)), axis image off title('LoG Response');

%% -----------------------------------------------------------
%% 2. K-Nearest Neighbour Classification
%% -----------------------------------------------------------

load fisheriris                            % built-in Iris data
feat   = meas(:,1:2);                      % use 2 features
labs   = categorical(species);

% Visualisation
figure(4)
gscatter(feat(:,1), feat(:,2), labs,'rgb','osd');
xlabel('Sepal length'); ylabel('Sepal width');
title('Iris Scatter Plot');

% Built-in KNN (all data)
knn1   = fitcknn(feat, labs);
pred1  = predict(knn1, feat);
acc1   = mean(labs == pred1);
fprintf('Resub accuracy (k=1): %.2f\n', acc1);

% Confusion matrix
figure(5), confusionchart(labs, pred1);
title(sprintf('Resub Accuracy = %.2f', acc1));

% Hold-out split (80/20)
trainIdx = [1:40 51:90 101:140];
testIdx  = setdiff(1:150, trainIdx);
knn2     = fitcknn(feat(trainIdx,:), labs(trainIdx));
pred2    = predict(knn2, feat(testIdx,:));
acc2     = mean(labs(testIdx) == pred2);

figure(6), confusionchart(labs(testIdx), pred2);
title(sprintf('Hold-out Accuracy = %.2f', acc2));

% Manual KNN (k=1)
k = 1;
predManual = categorical.empty(numel(testIdx),0);
for n = 1:numel(testIdx)
    q        = feat(testIdx(n),:);
    dists    = sum((feat(trainIdx,:)-q).^2, 2);
    [~,idx]  = sort(dists);
    predManual(n,1) = mode(labs(trainIdx(idx(1:k))));
end
accManual = mean(predManual == labs(testIdx));

figure(7), confusionchart(labs(testIdx), predManual);
title(sprintf('Manual KNN Accuracy = %.2f', accManual));

%% -----------------------------------------------------------
%% 3. Super-Pixel Clustering & Segmentation
%% -----------------------------------------------------------

dog   = double(imread('Dog.jpg'))/255;
[L, N] = superpixels(dog, 50);

% Boundary overlay
figure(8), imshow(imoverlay(dog, boundarymask(L), 'cyan')); axis image off;

% Region-averaged color image
avgImg = zeros(size(dog));
for i = 1:N
    mask = L == i;
    for c = 1:3
        chan = dog(:,:,c); 
        val  = mean(chan(mask));
        avgChan = avgImg(:,:,c); 
        avgChan(mask) = val; 
        avgImg(:,:,c) = avgChan;
    end
end
figure(9), imshow(avgImg); axis image off; title('Region-Averaged Image');

% Simple foreground mask (norm threshold)
maskFG = sqrt(sum(avgImg.^2,3)) < 0.8;
figure(10), imshow(maskFG); title('Foreground Mask');

% Split & save
figure(11)
subplot(1,2,1), imshow(dog.*maskFG),     title('Object A'), axis image off;
subplot(1,2,2), imshow(dog.*(~maskFG)),  title('Object B'), axis image off;
imwrite(dog.*maskFG,    'DogSplit1.png');
imwrite(dog.*(~maskFG), 'DogSplit2.png');

% Contour overlay
figure(12), imshow(dog), axis image off; hold on;
contour(maskFG,'b','LineWidth',1); hold off;
