%����subject��sample
%�����ع��������
function [U1,P1] = mlperctsix(pdata,plabel)
cdata1=pdata;
h1=size(cdata1,1);
%��һ�ν�ά����
U1(1,:)=mean(cdata1,1);%���о�ֵ
U1(2,:)=median(cdata1,1);%��λ��
U1(4,:)=std(cdata1,1);% �����׼�ȨֵΪ1��ά��Ϊ1��Ҳ���Ǽ����б�׼��
U1(5,:)=mad(cdata1);%ƽ������ƫ��
U1(6,:)=quantile(cdata1,0.75,1)-quantile(cdata1,0.25,1);%�ķ�λ��Χ
[val1,ind1]=sort(cdata1);
U1(3,:)=mean(val1(ceil(h1*0.2):ceil(h1*0.8),:),1);% 20%��β��ֵ
P1=plabel(1:6,:);
end