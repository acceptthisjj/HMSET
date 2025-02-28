%��Ȩ����������ǩ���� ���ձ�ǩ ������kn  ��ȡ���� ռʱ����
function [real_labelset] = KN_compute_label(labelset,k,key1,key2,weight) 
    count=size(labelset,1)/k;
    real_labelset=[];
    for i=1:count%����ÿ������
        per=labelset(1+(i-1)*k:i*k,:);
        per(1,:)=per(1,:).*weight;
        per(2:end,:)=per(2:end,:).*((1-weight)/(k-1));
         if sum(per)>=0.5
             real_label=key1;
         else
             real_label=key2;
        end
        real_labelset=[real_labelset;real_label];
    end
    
end