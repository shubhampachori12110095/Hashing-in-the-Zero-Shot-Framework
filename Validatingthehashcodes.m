 

for i = 1:size(hammTrain,1)
    aaaa{i} = find(hammTrain(i,:) == min(hammTrain(i,:)));
end


logicalarray = zeros(size(hammTrain,1),1);

for j = 1:size(logicalarray)
    logicalarray(j) = ismember(trainInstancesLabels(j),aaaa{j});
end
    

mean(logicalarray);