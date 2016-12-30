function [Z_nmlz, Z] = get_Zfortest(s,sigma2) 
%%
    load('parametersoflambdaandgamma.mat')
    load('binAttrClasses.mat') %  binAttributes mein jo values hai woh class ke numbers ke hisaab se hi hai. That is original class number hi serial number hai isme. Not any order according to the names.
    load('alpha_weightsforourcase.mat')
    load('KTest.mat')
    load('Stest.mat')
    load('realclasslabelsarrangedaccordingtodataset.mat')
    load('testlabelswithoriginalclasses.mat')
    
    
    % Use either attrClasses or Stest.
    
    pred = Stest*Alpha'*KTest'; % If you want to predict only with respect to the test classes.
    %pred = attrClasses*Alpha'*KTest'; % If you want to predict with respect to all the classes.
    pred = pred';
    
    testInstancesLabels = testInstancesLabels';
    %%
    %s = 10; %no. of classes you want with which we want the test output to be closest to. That is number of classes with which 
    
    n = size(pred,1);
    val = zeros(n,s);
    pos = val;

    %%
    % This gives the top s predictions for each testing instance.
    for i = 1:s 
        [val(:,i),pos(:,i)] = max(pred,[],2);
        tep = (pos(:,i)-1)*n+[1:n]'; % This tep changes the highest probability to the lowestt so that second maximum point could be detected. This follows till the top s anchors are detected.
        pred(tep) = 0; 
    end
    clear tep
    %% The following code calculates the accuracy of the algorithm. Whether the correct class is in the top s classes or not.
    classificationoutput = zeros(size(pred,1),1);
    for i = 1:length(classificationoutput)
         classificationoutput(i) = ismember(testInstancesLabels(i),pos(i,:));
         %classificationoutput(i) = ismember(testlabelswithoriginalclasses(i),originalclasstestoutput(i,:));   
    end   

    
    % Here r is the accuracy with which the class  
    r=mean(classificationoutput);
    
    m = size(Stest,1);
    %m = size(A,1); % If you want to predict with respect to the all classes that is 50 classes.
    if nargout >= 2
        Z = zeros(n,m);
        tep = (pos-1)*n+repmat([1:n]',1,s);
        Z([tep]) = [val];
        Z = sparse(Z);
    end
    val_nmlz = repmat(sum(val,2).^-1,1,s).*val;
    
    for i = 1:s
       val_nmlz(:,i) = (sigma2^(-(i-1)))*val_nmlz(:,i); 
    end    
    
    clear val;
    
    Z_nmlz = zeros(n,m);
    if ~exist('tep', 'var')
        tep = (pos-1)*n+repmat([1:n]',1,s);
    end
    Z_nmlz([tep]) = [val_nmlz];
    Z_nmlz = sparse(Z_nmlz);
    
    clear val_nmlz;
    clear tep;
    clear pos;
end