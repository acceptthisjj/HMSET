function [dim,direction,thresh,alpha] = adaBoostTrainDs(data,label,iter)
[m,~] = size(data);
% 初始化权值D
D = ones(m,1)/m;
alpha = zeros(iter,1);
% 记录T方向
direction = zeros(iter,1);
% 记录T属于哪一个维度
dim = zeros(iter,1);
% 初始化阈值T
thresh = zeros(iter,1);
for i = 1:iter
    [dim(i),direction(i),thresh(i),best_label,error] = ...
        buildSimpleStump(data,label,D);
    %计算alpha
    alpha(i) = 0.5*log((1-error)/max(error,1e-15));
    %更新权值D
    D = D.*(exp(-1*alpha(i)*(label.*best_label)));
    D = D/sum(D);
end