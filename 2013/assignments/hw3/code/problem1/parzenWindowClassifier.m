function prediction = parzenWindowClassifier (trainData, testData, hh)
% number of elements in the training set
N = size(trainData,1);
% dimension of the data
d = size(trainData,2);
% number of elements in the training set, such that C_k = y
n = 10; % it is 10 for all C_k

prediction = zeros(3,3);

for j = 1:3
    % P(C1|X)
    pXC1 = 0;
    for i = 1:10
        % P(C1|X)*P(X) = P(X|C1)*P(C1)
        pXC1 = pXC1 + (1/n)*(1/((2*pi*hh^2)^(d/2)))*exp(-((testData(j,:) - trainData(i,1:3))*(testData(j,:) - trainData(i,1:3))')/((2*hh)^2));
    end
    pXC2 = 0;
    for i = 11:20         
        pXC2 = pXC2 + (1/n)*(1/((2*pi*hh^2)^(d/2)))*exp(-((testData(j,:) - trainData(i,1:3))*(testData(j,:) - trainData(i,1:3))')/((2*hh)^2));
    end
    pXC3 = 0;
    for i = 21:30 
        pXC3 = pXC3 + (1/n)*(n/N)*(1/((2*pi*hh^2)^(d/2)))*exp(-((testData(j,:) - trainData(i,1:3))*(testData(j,:) - trainData(i,1:3))')/((2*hh)^2));
    end
    PX = (pXC1*(n/N) + pXC2*(n/N) + pXC3*(n/N));
    
    % each line of the prediction array states the values of the 
    % posteriors P(C1|x), P(C2|x), P(C3|x). Each column for each of the 
    % 3 points to be classified.
    prediction(:,j) = [pXC1*(n/N)/PX; pXC2*(n/N)/PX; pXC3*(n/N)/PX];
end

% extra code for plots for visualization of training dataset and 
% test points (just to check if the classification makes sense)

% % training data for P(X|Ck = 2)
% scatter3(trainData(11:20,1),trainData(11:20,2),trainData(11:20,3),'*','g'); hold on;
% % training data for P(X|Ck = 3)
% scatter3(trainData(21:30,1),trainData(21:30,2),trainData(21:30,3),'*','b'); hold on;
% 
% % test data
% scatter3(testData(1:3,1),testData(1:3,2),testData(1:3,3),'*','k'); hold off;

% training data for P(X|Ck = 1)
% hAx(1) = subplot(221); h = scatter3(trainData(1:10,1),trainData(1:10,2),trainData(1:10,3),'*','r');   view(3)
% hAx(2) = subplot(222); copyobj(h,hAx(2)); view(0,90), title('X-Y')
% hAx(3) = subplot(223); copyobj(h,hAx(3)); view(0,0) , title('X-Z')
% hAx(4) = subplot(224); copyobj(h,hAx(4)); view(90,0), title('Y-Z')
% 
% hold on;
% 
% % training data for P(X|Ck = 2)
% hAx(1) = subplot(221); h = scatter3(trainData(11:20,1),trainData(11:20,2),trainData(11:20,3),'*','g');   view(3)
% hAx(2) = subplot(222); copyobj(h,hAx(2)); view(0,90), title('X-Y')
% hAx(3) = subplot(223); copyobj(h,hAx(3)); view(0,0) , title('X-Z')
% hAx(4) = subplot(224); copyobj(h,hAx(4)); view(90,0), title('Y-Z')
% 
% % training data for P(X|Ck = 3)
% hAx(1) = subplot(221); h = scatter3(trainData(21:30,1),trainData(21:30,2),trainData(21:30,3),'*','b');   view(3)
% hAx(2) = subplot(222); copyobj(h,hAx(2)); view(0,90), title('X-Y')
% hAx(3) = subplot(223); copyobj(h,hAx(3)); view(0,0) , title('X-Z')
% hAx(4) = subplot(224); copyobj(h,hAx(4)); view(90,0), title('Y-Z')
% 
% % test data
% hAx(1) = subplot(221); h = scatter3(testData(1:3,1),testData(1:3,2),testData(1:3,3),'*','k');   view(3)
% hAx(2) = subplot(222); copyobj(h,hAx(2)); view(0,90), title('X-Y')
% hAx(3) = subplot(223); copyobj(h,hAx(3)); view(0,0) , title('X-Z')
% hAx(4) = subplot(224); copyobj(h,hAx(4)); view(90,0), title('Y-Z')
% 
% subplot(221);
% 
% hold off;

return;

