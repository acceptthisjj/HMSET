function [coeff,k]=mypca(data)
[coeff,score,latent,tsquared,explained,mu] = pca(data);
a=cumsum(latent)/sum(latent);%�ۻ�����ֵ
idx=find(a>0.95);
k=idx(1);
end