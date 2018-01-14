format long;
train_data = dlmread('data_bishop.txt', ' ');

dim = size(train_data, 2) -1;
train_dataX = train_data(:,dim);
train_dataY = train_data(:,dim+1);

plot (train_dataX, train_dataY,  'ob', 'LineWidth', 2);

test_dataX = [0:0.0001:1]';

[test_dataY] = poly_regression(train_dataX, train_dataY, test_dataX, 9);

hold on;
plot (test_dataX, test_dataY,  '.g', 'LineWidth', 1);

% print the axis labels
xlabel('x');
ylabel('y');

% hold off;

