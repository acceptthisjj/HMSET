function [res_X,res_Y,idx] = pick_neighbor2(X,Y,mu,n)
n_del = size(X,1)-n;
dist = sum(bsxfun(@power,bsxfun(@minus,X,mu),2),2); 
% 算数据点与平均值点距离
%bsxfun(@minus,X,mu) ；X数据集合中每行的中每一列减去mu中存放的每一列的平均值，得出每一个样本与样本平均值。
%bsxfun(@power,bsxfun(@minus,X,mu)；把上一步的得到的距离进行平方
% sum(上一步)；把上一步得到的矩阵按照行进行sum得到一个1列的矩阵，（每一行中的数据进行全部相加）
[val,idx] = sort(dist,'descend');%把距离从高到低排序，
idx = idx(1:n_del); %取出来距离最大的前6个
X(idx,:) = [];   %把训练数据与平均值相差 最 大的前6个置空，就是把与平均值差异最大的去掉。
Y(idx) = [];  %去差异最大的标签。把最大的前6个置空就是把与平均值差异最大的去掉。
res_X = X;   % 把去掉最大的，剩下的 数据集取出来
res_Y = Y; % 把去掉最大的，剩下的 数据集取出来
end