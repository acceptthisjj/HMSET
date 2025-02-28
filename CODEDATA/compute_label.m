%包络样本变换后 （a b c）标签计算 最终标签
function [real_labelset] = compute_label(labelset,perk,key1,key2) 
    count=size(labelset,1)/perk;
    real_labelset=[];
    for i=1:count
        per=labelset(1+(i-1)*perk:i*perk,:);
        p1=sum(per==key1);
        p2=sum(per==key2);
        if p1>=p2
            real_label=key1;
        else
            real_label=key2;
        end
        real_labelset=[real_labelset;real_label];
    end
end