%����subject��sample
%���طֺ�2����һ�α任ָ��2*6
function [U1,P1,U2,P2] = perctsix(pdata,plabel,c1,c2)
cdata1=pdata(c1,:);
cdata2=pdata(c2,:);

h1=size(cdata1,1);
h2=size(cdata2,1);
%��һ�ν�ά����
U1(1,:)=sum(cdata1,1);%���о�ֵ
U1(2,:)=median(cdata1,1);%��λ��
U1(4,:)=std(cdata1,1);% �����׼�ȨֵΪ1��ά��Ϊ1��Ҳ���Ǽ����б�׼��
U1(5,:)=mad(cdata1);%ƽ������ƫ��
U1(6,:)=quantile(cdata1,0.75,1)-quantile(cdata1,0.25,1);%�ķ�λ��Χ
[val1,ind1]=sort(cdata1);
U1(3,:)=mean(val1(ceil(h1*0.2):ceil(h1*0.8),:),1);% 20%��β��ֵ

U2(1,:)=sum(cdata2,1);%���о�ֵ
U2(2,:)=median(cdata2,1);%��λ��
U2(4,:)=std(cdata2,1);% �����׼�ȨֵΪ1��ά��Ϊ1��Ҳ���Ǽ����б�׼��
U2(5,:)=mad(cdata2);%ƽ������ƫ��
U2(6,:)=quantile(cdata2,0.75,1)-quantile(cdata2,0.25,1);%�ķ�λ��Χ
[val2,ind2]=sort(cdata2);
U2(3,:)=mean(val2(ceil(h2*0.2):ceil(h2*0.8),:),1);% 20%��β��ֵ

tempY=[];
for i=1:6
        tempY=[tempY;plabel(1,:)];
end
P1=tempY;
P2=tempY;
end