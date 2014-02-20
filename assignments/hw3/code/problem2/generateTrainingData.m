function data = generateTrainingData (N, a1, a2)

% mean and variance values for the [x1, x2] and [z] standard normal 
% random variables, i.e. with parameters [u = 0; var = 1].
% in this case, we can use the expression u + var.*randn(N,1), i.e.
% randn(N,1)
%u = 0.0; var = 1.0;

% generate n instances of [x1, x2] and [z]
z = randn(N,1);

% merge x1 and x2 into a (n x 2) matrix
x = [randn(1,N);randn(1,N)]';

% parameters [a, c] for the sigmoid function in MATLAB (sigmf(x, [a, c]))
%[a, c] = [1, 0];

% generate the training data
%s = arrayfun(@(x) sigmf(x, [1, 0]), (x*a1));

data = zeros(N,4);

for n = 1:N
    data(n,:) = [x(n,1), x(n,2), z(n), sigmf(x(n,:)*a1, [1, 0]) + (x(n,:)*a2).^2 + 0.30*z(n)];
end

return