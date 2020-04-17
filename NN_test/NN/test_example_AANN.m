function [nn,mu,sigma]=test_example_AANN(train_x, n_hidden1, n_hidden2)


%  train_x is in the form of [samples,variables]
%  nn is the trained neural network
% mu, sigma are the zscore values


dimensions=size(train_x);
n_samples=dimensions(1);

n_input=dimensions(2);

% normalize
[train_x, mu, sigma] = zscore(train_x);
%test_x = normalize(test_x, mu, sigma);
 train_x = rescale(train_x);
 
 train_y=(train_x+1)/2;



% 
%% training AANN with sigmoid activation function
%% the following parameters have to be set up
rand('state',0)

%nn = nnsetup([220 100 5 100 220]);

nn = nnsetup([n_input n_hidden1 n_hidden2 n_hidden1 n_input]);



nn.activation_function = 'sigm';    %  Sigmoid activation function
nn.learningRate = 0.01;                
opts.numepochs =  200;               %  Number of full sweeps through data

nba=65535/opts.numepochs;
batchsize=uint16(n_samples/nba);


opts.batchsize = 500;                %  Take a mean gradient step over this many samples
opts.plot  = 1;                     %  enable plotting   





nn = nntrain(nn, train_x, train_y, opts);

%[er, bad] = nntest(nn, test_x, test_y);

