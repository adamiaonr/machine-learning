% generate training and test sets, composed by 600 samples each
% use a specific mix of gaussians to generate the samples
px_c1 = gmdistribution([0 0; 4 0; 2 -2], cat(3, [1 0; 0 1], [1 0; 0 1], [1 0; 0 1]), ones(1,3)/2);
px_c2 = gmdistribution([0 4; 2 2; 4 4], cat(3, [1 0; 0 1], [1 0; 0 1], [1 0; 0 1]), ones(1,3)/2);

% don't forget the label C_k in the 3rd column (C_1 = 1, C_2 = 2)
trainingData = [random(px_c1,300), ones(300,1); random(px_c2,300), 2.*ones(300,1)];
testData = [random(px_c1,300), ones(300,1); random(px_c2,300), 2.*ones(300,1)];

% couldn't use gmdistribution() and random() due to statistics toolbox 
% licensing problems, attempted to reproduce behavior via randn(), assuming
% a covariance of [1 0; 0 1]
% trainingData = [(1/3).*(randn(300, 2) + [4.*ones(300,1) zeros(300,1)] + randn(300, 2) + [2.*ones(300,1) -2.*ones(300,1)] + randn(300, 2)), ones(300,1);
%     (1/3).*([zeros(300,1) 4.*ones(300,1)] + randn(300, 2) + [2.*ones(300,1) 2.*ones(300,1)] + randn(300, 2) + [4.*ones(300,1) 4.*ones(300,1)] + randn(300, 2)), 2.*ones(300,1)];
% 
% testData = [(1/3).*(randn(300, 2) + [4.*ones(300,1) zeros(300,1)] + randn(300, 2) + [2.*ones(300,1) -2.*ones(300,1)] + randn(300, 2)), ones(300,1);
%     (1/3).*([zeros(300,1) 4.*ones(300,1)] + randn(300, 2) + [2.*ones(300,1) 2.*ones(300,1)] + randn(300, 2) + [4.*ones(300,1) 4.*ones(300,1)] + randn(300, 2)), 2.*ones(300,1)];


% run the knnclassifier, for values of k = 10, 20, 30 and 40
predictions = zeros(size(testData,1), 4, 4);

predictions(:,:,1) = knnclassifier(trainingData, testData, 0.1, 10);
predictions(:,:,2) = knnclassifier(trainingData, testData, 0.1, 20);
predictions(:,:,3) = knnclassifier(trainingData, testData, 0.1, 30);
predictions(:,:,4) = knnclassifier(trainingData, testData, 0.1, 40);

% calculate the divergences for each prediction (k)

dPP = zeros(4,2);
N = size(testData, 1);
M = 2;

for n = 1:N
    
    % test point x
    x1 = testData(n,1);
    x2 = testData(n,2);

    % class conditional probabilities for test point x
    PXC1 = (1/(3*2*pi))*(exp(-0.5*(x1.^2 + x2.^2)) + exp(-0.5*((x1 - 4).^2 + x2.^2)) + exp(-0.5*((x1 - 2).^2 + (x2 + 2).^2)));
    PXC2 = (1/(3*2*pi))*(exp(-0.5*(x1.^2 + (x2 - 4).^2)) + exp(-0.5*((x1 - 2).^2 + (x2 - 2).^2)) + exp(-0.5*((x1 - 4).^2 + (x2 - 4).^2)));

    % actual posterior probabilities for test point x
    PC1X = (PXC1 * (1/2))/(PXC1 * (1/2) + PXC2 * (1/2));
    PC2X = (PXC2 * (1/2))/(PXC1 * (1/2) + PXC2 * (1/2));
    
    % for equation 1, k = 10
    dPP(1,1) = dPP(1,1) + PC1X * log(PC1X/predictions(n,3,1)) + PC2X * log(PC2X/predictions(n,4,1));
    
    % for equation 3, k = 10
    dPP(1,2) = dPP(1,2) + PC1X * log(PC1X/predictions(n,1,1)) + PC2X * log(PC2X/predictions(n,2,1));
    
    % for equation 1, k = 20
    dPP(2,1) = dPP(2,1) + PC1X * log(PC1X/predictions(n,3,2)) + PC2X * log(PC2X/predictions(n,4,2));
    
    % for equation 3, k = 20
    dPP(2,2) = dPP(2,2) + PC1X * log(PC1X/predictions(n,1,2)) + PC2X * log(PC2X/predictions(n,2,2));

    % for equation 1, k = 30
    dPP(3,1) = dPP(3,1) + PC1X * log(PC1X/predictions(n,3,3)) + PC2X * log(PC2X/predictions(n,4,3));
    
    % for equation 3, k = 30
    dPP(3,2) = dPP(3,2) + PC1X * log(PC1X/predictions(n,1,3)) + PC2X * log(PC2X/predictions(n,2,3));

    % for equation 1, k = 40
    dPP(4,1) = dPP(4,1) + PC1X * log(PC1X/predictions(n,3,4)) + PC2X * log(PC2X/predictions(n,4,4));
    
    % for equation 3, k = 40
    dPP(4,2) = dPP(4,2) + PC1X * log(PC1X/predictions(n,1,4)) + PC2X * log(PC2X/predictions(n,2,4));

end

dPP = (1/N).*dPP
 

