%���������任�� ��a b c����ǩ���� ���ձ�ǩ
%���ǩ
function [real_labelset] = B_compute_label(labelset,perk) 
    count=size(labelset,1)/perk;
    real_labelset=[];
    for i=1:count
        per=labelset(1+(i-1)*perk:i*perk,:);%�����ǩ
        real_label=mode(per);
        real_labelset=[real_labelset;real_label];
    end
end