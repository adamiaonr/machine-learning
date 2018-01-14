% sizes of training and test sets specified in the execises
N = 200;
T = 1000;

% generate training and test data for training and testing the neural
% network
y = generateTrainingData(N, [3; 3], [3; -3]);
t = generateTrainingData(T, [3; 3], [3; -3]);

% plot the training data
% hAx(1) = subplot(221); h = scatter3(y(1:N,1), y(1:N,2), y(1:N,4),'*','r');   view(3) 
% hAx(2) = subplot(222); copyobj(h,hAx(2)); view(0,90), title('X-Y')
% hAx(3) = subplot(223); copyobj(h,hAx(3)); view(0,0) , title('X-Z')
% hAx(4) = subplot(224); copyobj(h,hAx(4)); view(90,0), title('Y-Z')

% train and test the neural network with sets generated above (plots of 
% errors are generated inside the function).
new_y = trainAndTestNeuralNet(y,t);

