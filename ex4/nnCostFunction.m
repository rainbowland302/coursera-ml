function [J grad] = nnCostFunction(nn_params, ...
                                   input_layer_size, ...
                                   hidden_layer_size, ...
                                   num_labels, ...
                                   X, y, lambda)
%NNCOSTFUNCTION Implements the neural network cost function for a two layer
%neural network which performs classification
%   [J grad] = NNCOSTFUNCTON(nn_params, hidden_layer_size, num_labels, ...
%   X, y, lambda) computes the cost and gradient of the neural network. The
%   parameters for the neural network are "unrolled" into the vector
%   nn_params and need to be converted back into the weight matrices.
%
%   The returned parameter grad should be a "unrolled" vector of the
%   partial derivatives of the neural network.
%

% Reshape nn_params back into the parameters Theta1 and Theta2, the weight matrices
% for our 2 layer neural network
Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
                 hidden_layer_size, (input_layer_size + 1));

Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
                 num_labels, (hidden_layer_size + 1));

% Setup some useful variables
m = size(X, 1);

% You need to return the following variables correctly
J = 0;
Theta1_grad = zeros(size(Theta1));
Theta2_grad = zeros(size(Theta2));

% ====================== YOUR CODE HERE ======================
% Instructions: You should complete the code by working through the
%               following parts.
%
% Part 1: Feedforward the neural network and return the cost in the
%         variable J. After implementing Part 1, you can verify that your
%         cost function computation is correct by verifying the cost
%         computed in ex4.m
%
% Part 2: Implement the backpropagation algorithm to compute the gradients
%         Theta1_grad and Theta2_grad. You should return the partial derivatives of
%         the cost function with respect to Theta1 and Theta2 in Theta1_grad and
%         Theta2_grad, respectively. After implementing Part 2, you can check
%         that your implementation is correct by running checkNNGradients
%
%         Note: The vector y passed into the function is a vector of labels
%               containing values from 1..K. You need to map this vector into a
%               binary vector of 1's and 0's to be used with the neural network
%               cost function.
%
%         Hint: We recommend implementing backpropagation using a for-loop
%               over the training examples if you are implementing it for the
%               first time.
%
% Part 3: Implement regularization with the cost function and gradients.
%
%         Hint: You can implement this around the code for
%               backpropagation. That is, you can compute the gradients for
%               the regularization separately and then add them to Theta1_grad
%               and Theta2_grad from Part 2.
%
% y 5000 * 1
% X 5000 * 400
% Theta1 25 * 401
% Theta2 10 * 26
% p 5000 * 1
%a2 = sigmoid([ones(m,1) X] * Theta1') % 5000 * 25
%a3 = sigmoid([ones(m,1) a2] * Theta2' )
%[v ,p] = max(a3, [], 2);

a2 = sigmoid( [ones(m,1) X] * Theta1' );
% Theta1_J = 1 / m * ( (log(a2)' * -y) - log(1 - a2)' * (1 - y) ) + lambda / (2 * m) * Theta1(2:end, 2:end) * Theta1(2:end, 2:end)';
a3 = sigmoid( [ones(m,1) a2] * Theta2' );
% a3 5000 * 10
% y 5000 * 1
for i = 1:m;
    shapeY = zeros(num_labels, 1);
    shapeY(y(i)) = 1;
    fprintf('y(i) %f\n', y(i));
    fprintf('shapeY %f\n', shapeY);
    %[value, index] = max(a3(i,:), [], 2);
    vector_J(i) = ( (log(a3(i,:)) * -shapeY) - log(1 - a3(i,:)) * (1 - shapeY) );
end;
%J = 1 / m * sum(vector_J)
%regulization: lambda / (2 * m) * theta(2:end)' * theta(2:end);

for i = 1:hidden_layer_size;
    reg1Matrix = lambda / (2 * m) * Theta1(:, 2:end) .* Theta1(:, 2:end);
end;
reg1 = sum(sum(reg1Matrix));
for i = 1:num_labels;
    reg2Matrix = lambda / (2 * m) * Theta2(:, 2:end) .* Theta2(:, 2:end);
end;
reg2 = sum(sum(reg2Matrix));

J = 1 / m * sum(vector_J) + reg1 + reg2;
%% the J is correct , the block below should be examine

Delta2 = 0;
Delta1 = 0;
for i = 1:m;
    x = [1; X(i, :)']; % 401 * 1

    z2 = [Theta1 * x ]; % 25 * 1

    a2 = [1; sigmoid(z2)];

    z3 = [Theta2 * a2]; % 10 * 1

    a3 = sigmoid(z3);


    shapeY = zeros(num_labels, 1);
    shapeY(y(i)) = 1;
    delta3 = a3 - shapeY; % 10 * 1
    fprintf('y(i) %f\n', y(i));
    fprintf('shapeY %f\n', shapeY);
    delta2 = Theta2' * delta3 .* [1; sigmoidGradient(z2)]; % 26 * 1


    Delta2 = Delta2 + delta3 * a2'; % 10 * 26
    fprintf('Delta2 %f\n', sum(sum(Delta2)));
    Delta1 = Delta1 + delta2(2:end) * x'; % x = a1 25 * 401
    fprintf('Delta1 %f\n', sum(sum(Delta1)));
end

%Theta1_grad = 1 / m *  X' * ( h - y ) + lambda / m * Theta1(:, 2:end)';
%Theta1_grad(1) = 1 / m *  X( : , 1)' * ( h - y ); % theta - alpha * grad

%Theta2_grad = 1 / m *  X' * ( h - y ) + lambda / m * Theta2(:, 2:end)';
%Theta2_grad(1) = 1 / m *  X( : , 1)' * ( h - y ); % theta - alpha * grad
% -------------------------------------------------------------

% =========================================================================

% Unroll gradients
Theta2_grad = Delta2 / m; % 10 * 26
Theta1_grad = Delta1 / m; % 25 * 401

grad = [Theta1_grad(:) ; Theta2_grad(:)];

end
