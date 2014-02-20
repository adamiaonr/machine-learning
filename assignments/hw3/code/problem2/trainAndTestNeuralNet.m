function prediction = trainAndTestNeuralNet(trainingData, testData)

% regularization factor (set to 0 in order to test overfitting)
%lambda = exp(-18);
lambda = 0;

% number of single pass through the entire training set, followed by 
% testing the test data
n_epochs = 500;

% learning rate (alpha)
alpha = 0.001;

% initialize the weight arrays to random values in the interval [-0.5; 0.5]
rand('state',sum(100*clock));
w1 = -0.5 + rand(10, 3);
w2 = -0.5 + rand(11, 1);

% initialize activations from for hidden layer
a = zeros(10,1);
at = zeros(10,1);

% initialize the outputs of hidden layer
z = zeros(10,1);
zt = zeros(10,1);

% initialize deltas (delta_1 for the hidden units, delta_2 for the 
% output unit)
delta_1 = zeros(10,1);
delta_2 = 0;

% size of training and test data
n_training = size(trainingData,1);
n_test = size(testData,1);

% arrays for keeping error values
% error calculated as sum-of-squares error
test_error = zeros(n_epochs,1);
training_error = zeros(n_epochs,1);

% predictions array
prediction = zeros(n_test,1);

% run n_epochs training-test iterations
for e = 1:n_epochs
    
    % i've included the test phase first in an iteration (test with initial
    % random weights)
    
    % evaluate test data error
    error = 0;
    
    for t = 1:n_test
        
        % activations at(est), note the extension of array testData(t,:) 
        % with a 1, to account for the bias in the hidden units.
        at = w1*[1, testData(t,1:2)]';
        
        % output of hidden units
        for n = 1:10
            
            % logistic sigmoid function is used in the hidden layers
            zt(n) = 1 / (1 + exp(-at(n,:)));
        end
        
        % output unit values yt, note that output unit is linear,
        % activation function h(.) is identity
        yt = w2'*[1; zt];
        
        prediction(t) = yt;
        
        % evaluate the error (sum-of-squares)
        error = error + 0.5 * (yt - testData(t,4)).^2;

    end
    
    % save the test data error;
    test_error(e) = error;
    
    % training phase
    error = 0;
    
    for t = 1:n_training

        % activations a for hidden units, note the extension of array 
        % testData(t,:) with a 1, to account for the bias in the hidden 
        % units.
        a = w1*[1, trainingData(t,1:2)]';

        % output of hidden units
        for n = 1:10
            
            % logistic sigmoid function is used as activation function in 
            % the hidden layers
            z(n) = 1 / (1 + exp(-a(n,:)));
        end

        % activation of output unit
        y = w2'*[1; z];

        % evaluate delta of output unit
        delta_2 = trainingData(t,4) - y;
                
        % evaluate the training error (also sum-of-squares)
        error = error + 0.5 * (y - trainingData(t,4)).^2;

        % backpropagation of delta, obtainign delta for hidden units, note
        % the derivative of the activation function (logistic sigmoid), 
        % z.*(1 - z)
        delta_1 = (delta_2 .* diag(w2(2:11))) * z.*(1 - z);
        
        % evaluate derivatives of dE/dW (with regularization factor lambda)
        dEdw1 = delta_1 * [1, trainingData(t,1:2)] + lambda .* w1;
        dEdw2 = delta_2 * [1; z] + lambda .* w2;

        % update weights
        w1 = w1 + alpha * dEdw1;
        w2 = w2 + alpha * dEdw2;
                
    end
    
    % save the training data error
    training_error(e) = error;

end


subplot(121); plot(1:n_epochs, training_error, 'r.'); title('Error (training) vs. Epochs'); xlabel('N Epochs'); ylabel('Sum-of-Squares Error');
subplot(122); plot(1:n_epochs, test_error, 'b.'); title('Error (test) vs. Epochs'); xlabel('N Epochs'); ylabel('Sum-of-Squares Error');

return