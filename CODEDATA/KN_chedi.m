%计算每个样本欧式最近k个样本（探寻点）并计算最短测地距离路径
%返回测地距离图
% step1: Calculate the k nearest distance 
function [D]=KN_chedi(X,k2)
[m, ~] = size(X);
D = zeros(m);
%设定可直达的改点最近点（该节点的可移动步调）
k1=k2;
for i =1 : m
    xx = repmat(X(i, :), m, 1);
    diff = xx - X;
    dist = sum(diff.* diff, 2);%计算该样本与其他样本距离
    [dd, pos] = sort(dist);
    index = pos(1 : k1 + 1);
    index2 = pos(k1 + 2 : m);
    D(i,index) = sqrt(dist(index));
    D(i,index2) = inf;
end
%step2: recalculate shortest distant matrix
for k=1:m
    for i=1:m
        for j=1:m
            if D(i,j)>D(i,k)+D(k,j)
                D(i,j)=D(i,k)+D(k,j);
            end
        end
    end
end
end