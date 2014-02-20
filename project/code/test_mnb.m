% load the train and test datasets for the 20 Newsgroups detaset, available
% in http://qwone.com/~jason/20Newsgroups/
% i've made the dataset used in my MATLAB implementation in 
% http://paginas.fe.up.pt/~up200400437/dataset.zip

% create sparse matrices for the *.data datasets.
% train_data = spconvert(load('/home/adamiaonr/Dropbox/Workbench/PhD/machine learning/project/experiments/dataset/train.data'));
% test_data = spconvert(load('/home/adamiaonr/Dropbox/Workbench/PhD/machine learning/project/experiments/dataset/test.data'));
% 
% train_class = load('/home/adamiaonr/Dropbox/Workbench/PhD/machine learning/project/experiments/dataset/train.label');
% test_class = load('/home/adamiaonr/Dropbox/Workbench/PhD/machine learning/project/experiments/dataset/test.label');

% train_data = spconvert(load('/home/adamiaonr/Documents/PhD/coursework/machine learning/train.data'));
% test_data = spconvert(load('/home/adamiaonr/Documents/PhD/coursework/machine learning/test.data'));
% 
% train_class = load('/home/adamiaonr/Documents/PhD/coursework/machine learning/train.label');
% test_class = load('/home/adamiaonr/Documents/PhD/coursework/machine learning/test.label');

% set the values of alpha for the training (in this case we set alpha = 1,
% Laplace smoothing, as used in Nigam2000).
alpha = 1;

% size of the dictionary D
D = max([size(train_data, 2); size(test_data, 2)]);

T = 20;

A_l = [20, 40, 100, 500, 1000, 5000, 7520];
N = size(train_data,1);

% num_runs
num_runs = 3;

for k = 1:size(A_l, 2)

    A_l_t = A_l(k) / T;

    super_accuracy = 0;
    
    % train a MNB text classifier
    for i = 1:num_runs

        % topic offset
        offset = 0;

        % the I array will be a mask which indicates which indices of train_data 
        % will be used as labeled / unlabeled training datasets. 
        I = zeros(N, 1);

        for t = 1:T

            % find the number of examples of articles of topic t.
            A_t = sum((train_class == t));

            index = randperm(A_t, A_l_t);
            index = bsxfun(@plus, offset, index);

            % set the computed indexes to 1.
            I(index) = 1;

            % update the topic offset.
            offset = offset + A_t;
        end


        [theta_w, theta_c] = mnb_train(train_data((I == 1),:), train_class((I == 1),:), D, alpha, 'false');
        %[theta_w, theta_c] = mnb_train(train_data, train_class, D, alpha, 'false');

        % run the classifier on the test data
        classification = mnb_class(log(norm_p_w_t(theta_w,alpha)), log(norm_p_t(theta_c,alpha)), test_data, 1000);
        %classification = mnb_class(log(theta_w), log(theta_c), test_data, size(test_data, 1));

        % compute the fully-supervised accuracy
        [maxs, index] = max(classification, [], 2);
        super_accuracy = super_accuracy + (sum(((index - test_class) == 0)) / (size(test_class, 1)));
    end

    A_l(k)
    avg = super_accuracy / num_runs
        
end
