function [labelarr]=Subspace2_UseC4_5(numtree,trainX,trainY,testX,envelope)
%��ö���ز����ӿռ�ѵ��������������ò��Ա�ǩ
%�������80%���ӿռ�
%���ӿռ�Ԥ����
labelarr=[];

[b1,m1,n1]=unique(trainX,'rows','stable'); %���Զ�����  ��������� ѡ�����������
%�޳��ظ�
dataX=b1;
dataY=trainY(m1,:);

m=size(dataY,1);
for i=1:numtree
    A=randperm(m,round(m*0.8));
    subtrainx= dataX(A,:);
    subtrainY= dataY(A,:);
    prey=Use_C4_5(subtrainx',subtrainY',testX',5,10);
    prey=prey';%һ�б�һ��
    re_prey=B_compute_label(prey,envelope);%������Ԥ���ǩת��ԭ����Ԥ���ǩ
    labelarr=[labelarr,re_prey];
end
end