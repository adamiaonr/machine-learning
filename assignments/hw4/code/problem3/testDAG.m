function [] = testDAG()
	close all;
	rmpath('/home/adamiaonr/Dropbox/Workbench/PhD/machine learning/homework/HW4/SVMs/libsvm-mat-2.8-1');
	addpath ('/home/adamiaonr/Dropbox/Workbench/PhD/machine learning/homework/HW4/SVMs/libsvm-mat-2.8-1');
	rmpath('mysvm');
	addpath ('mysvm');
	rmpath('mysvm/common');
	addpath ('mysvm/common');    

    data = dlmread('testData.txt', ';');
	data = data(1:100,:); %CHANGE HERE 
	size(data)
	trainSetClass = data(:,end);
	trainSetFeatures = data(:,1:end-1);
	K = max(trainSetClass)
    
    figure;
    hold on;
    line= ['ob'; '*g'; '+c'; 'xr'; '>y'];
    for k=1:K
        idx = find (trainSetClass == k);
        plot (trainSetFeatures(idx,1),trainSetFeatures(idx,2),  line(k,:), 'LineWidth', 2);
    end
    hold off;


    [X,Y] = meshgrid(0:.01:1,0:.01:1);
    test_data_X = [reshape(X, numel(X), 1) reshape(Y, numel(Y) , 1)];
    
 	Cvalue = 1;       

    disp('LIBSVM')
    %FOR COMPARISON ONLY: using libsvm as a multiclass classifier
	configStr = sprintf('-s 0 -t 1 -d 2 -r 1 -g 1 -c %d', Cvalue);
	net = svmtrain(trainSetClass, trainSetFeatures, configStr);
   	testSetClass = ones(size(test_data_X,1), 1); %libSVM needs but does not really uses 
	[test_data_Y1] = svmpredict (testSetClass, test_data_X, net);

    disp ('SVM_DAG')
    [test_data_Y2] = SVM_DAG (trainSetFeatures, trainSetClass, test_data_X, Cvalue);
            
   
    figure;    
    hold on;
    line= ['ob'; '*g'; '+c'; 'xr'; '>y'];
    for k=1:K
        idx = find (test_data_Y1 == k);
        plot (test_data_X(idx,1), test_data_X(idx,2),  line(k,:), 'LineWidth', 1);
    end
    hold off;

    figure
    hold on;
    line= ['ob'; '*g'; '+c'; 'xr'; '>y'];
    for k=1:K
        idx = find (test_data_Y2 == k);
        plot (test_data_X(idx,1), test_data_X(idx,2),  line(k,:), 'LineWidth', 1);
    end
    hold off;
    
return