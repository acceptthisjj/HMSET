
function [D]=KN_chediT(testData,trainData,k2)
[m, ~] = size(testData);
[n,~]=size(trainData);
D = zeros(m,n);
%设定可直达的改点最近点（该节点的可移动步调）
k1=k2;
for i =1 : m
    xx = trainData;
    diff = xx - testData(i,:);
    dist = sum(diff.* diff, 2);%计算该样本与其他样本距离
    [dd, pos] = sort(dist);
    index = pos(1 : k1 + 1);
    index2 = pos(k1 + 2 : n);
    D(i,index) = sqrt(dist(index));
    D(i,index2) = inf;
end

D1 = zeros(n);
for i =1 :n
    xx = trainData;
    diff = xx - trainData(i,:);
    dist = sum(diff.* diff, 2);%计算该样本与其他样本距离
    [dd, pos] = sort(dist);
    index = pos(1 : k1 + 1);
    index2 = pos(k1 + 2 : n);
    D1(i,index) = sqrt(dist(index));
    D1(i,index2) = inf;
end
%step2: recalculate shortest distant matrix

for i=1:m
    for k=1:n
        for j=1:n
            if D(i,j)>D(i,k)+D1(k,j)
                D(i,j)=D(i,k)+D1(k,j);
            end
        end
    end        
end

end