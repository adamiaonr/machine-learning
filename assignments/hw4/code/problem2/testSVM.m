function [] = testSVM()
	close all;
	rmpath('/home/adamiaonr/Dropbox/Workbench/PhD/machine learning/homework/HW4/SVMs/libsvm-mat-2.91-1');
	addpath ('/home/adamiaonr/Dropbox/Workbench/PhD/machine learning/homework/HW4/SVMs/libsvm-mat-2.91-1');
	data = dlmread('classification.txt', ' ');
	data = data(1:100,:); %CHANGE HERE 
	size(data)
	trainSetClass = data(:,end);
	trainSetFeatures = data(:,1:end-1);
	
	figure;
	hold on;
	idx = find (trainSetClass == 0);
	plot (trainSetFeatures(idx,1),trainSetFeatures(idx,2),  'or', 'LineWidth', 2);
	idx = find (trainSetClass == 1);
	plot (trainSetFeatures(idx,1),trainSetFeatures(idx,2),  '*b', 'LineWidth', 2);
	hold off;

	figure;
	x1 = -2.5:0.01:2.5;
	x2 = -2.5:0.01:2.5;
	
	[x11,x22] = meshgrid(x1, x2);
	test_data_X = [x11(:) x22(:)];
	testSetClass = ones(size(test_data_X,1), 1); %libSVM needs but does not really uses 
	
	Cvalue = 10;
	configStr = sprintf('-s 0 -t 1 -d 2 -r 1 -g 1 -c %d', Cvalue);
	%net = svmtrain(trainSetClass, trainSetFeatures, configStr);	
	%[test_data_Y, accuracy] = svmpredict (testSetClass, test_data_X, net);
	test_data_Y = basicSVM (trainSetFeatures, trainSetClass, test_data_X, Cvalue);
    
	x1x2out = reshape(test_data_Y, [length(x1) length(x2)]);
	hold on;
    contour(x11, x22, x1x2out, [-0.99 -0.99], 'b-'); %margin
	contour(x11, x22, x1x2out, [0 0], 'k-');	%boundary	
    contour(x11, x22, x1x2out, [0.99 0.99], 'g-'); 	%margin
	hold off;
	
	figure
	hold on;
	idx = find (test_data_Y == 0);
	plot (test_data_X(idx,1), test_data_X(idx,2),  'or', 'LineWidth', 1);
	idx = find (test_data_Y == 1);
	plot (test_data_X(idx,1), test_data_X(idx,2),  '*b', 'LineWidth', 1);	
	hold off;	

return