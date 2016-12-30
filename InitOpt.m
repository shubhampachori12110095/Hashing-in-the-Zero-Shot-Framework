function options = InitOpt(method)

options = [];
switch method
    
    case 'IMH-tSNE'
        options.s = 5;
        options.sigma = 0;
        options.sigma2 = 10;
  
    case 'IMH-LE'
        options.s = 5;
        options.sigma = 0;
        options.k = 20; % knn graph   
        options.bTrueKNN  = 0; % if 0, construct symmetric graph for S
        options.sigma2 = 10; % This sigma makes the weights weighted. That is it will weigh the weights because the top s five weights detected were of same
        %value nearly. Therefore it was necessary to decrease them with
        %some exponenetial power factor.
   

end
