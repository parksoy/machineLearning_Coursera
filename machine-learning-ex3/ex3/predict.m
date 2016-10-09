function p = predict(Theta1, Theta2, X)
%PREDICT Predict the label of an input given a trained neural network
%   p = PREDICT(Theta1, Theta2, X) outputs the predicted label of X given the
%   trained weights of a neural network (Theta1, Theta2)

% Useful values
%X=5000x400
%theta1 25x401
%theta2 10x26

m = size(X, 1); %5000 examples

num_labels = size(Theta2, 1); %10

% You need to return the following variables correctly 
p = zeros(size(X, 1), 1);

% ====================== YOUR CODE HERE ======================
% Instructions: Complete the following code to make predictions using
%               your learned neural network. You should set p to a 
%               vector containing labels between 1 to num_labels.
%
% Hint: The max function might come in useful. In particular, the max
%       function can also return the index of the max element, for more
%       information see 'help max'. If your examples are in rows, then, you
%       can use max(A, [], 2) to obtain the max for each row.
%



%input layer
a1=[ones(m,1), X]; %5000 examples, resulting in 5000x401, 401 units


%Hidden layer
%theta1 25x401,
z2=a1*Theta1';%theta1 25x401, resulting in 5000x25
a2=sigmoid(z2);%25units 5000X25
a2=[ones(m,1),a2]; %Make 5000x26

%output layer
z3=a2*Theta2';%5000x26 * theta2 10x26
a3=sigmoid(z3); %5000X10, 10 output units

h=a3; %5000X10 


[prob,index]=max(h,[],2);

%prob
%index
p=index; %5000X1

% =========================================================================


end
