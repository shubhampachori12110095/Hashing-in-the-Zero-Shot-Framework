function [E] = tSNEH(train_X, options)

% Set parameters
no_dims = options.maxbits;
init_dims = max(20, no_dims);
perplexity = 10;
% Run tSNE
[E] = tsne(train_X, [], no_dims, init_dims, perplexity);
% [E] = compute_mapping(train_X, 'tSNE', no_dims);%, init_dims, perplexity);


% ydata = tsne(X, labels, no_dims, initial_dims, perplexity)
%TSNE Performs symmetric t-SNE on dataset X
%
%   mappedX = tsne(X, labels, no_dims, initial_dims, perplexity)
%   mappedX = tsne(X, labels, initial_solution, perplexity)
%
% The function performs symmetric t-SNE on the NxD dataset X to reduce its 
% dimensionality to no_dims dimensions (default = 2). The data is 
% preprocessed using PCA, reducing the dimensionality to initial_dims 
% dimensions (default = 30). Alternatively, an initial solution obtained 
% from an other dimensionality reduction technique may be specified in 
% initial_solution. The perplexity of the Gaussian kernel that is employed 
% can be specified through perplexity (default = 30). The labels of the
% data are not used by t-SNE itself, however, they are used to color
% intermediate plots. Please provide an empty labels matrix [] if you
% don't want to plot results during the optimization.
% The low-dimensional data representation is returned in mappedX.


end
