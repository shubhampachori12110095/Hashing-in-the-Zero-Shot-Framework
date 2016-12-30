
SS = zeros(size(datasetfeatures_AwA,1),size(datasetfeatures_AwA,1));

for i = size(datasetfeatures_AwA,1)
    for j = size(datasetfeatures_AwA,1)
        SS(i,j) = dot(datasetfeatures_AwA(i,:),datasetfeatures_AwA(j,:));
    end
end    