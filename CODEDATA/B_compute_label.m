%包络样本变换后 （a b c）标签计算 最终标签
%多标签
function [real_labelset] = B_compute_label(labelset,perk) 
    count=size(labelset,1)/perk;
    real_labelset=[];
    for i=1:count
        per=labelset(1+(i-1)*perk:i*perk,:);%包络标签
        real_label=mode(per);
        real_labelset=[real_labelset;real_label];
    end
end