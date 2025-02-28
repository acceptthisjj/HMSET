function [labelval,labeltest]=C_subspace2_C4_5(numtree,trainX,trainY,validX,testX,envelope)
%��ö���ز����ӿռ�ѵ��������������ò��Ա�ǩ
%�������80%���ӿռ�
%���ӿռ�Ԥ����
%�ֱ����֤���Ͳ��Լ�
labelval=[];
labeltest=[];
[b1,m1,n1]=unique(trainX,'rows','stable'); %���Զ�����  ��������� ѡ�����������
%�޳��ظ�
dataX=b1;
dataY=trainY(m1,:);

m=size(dataY,1);
for i=1:numtree
    A=randperm(m,round(m*0.8));
    subtrainx= dataX(A,:);
    subtrainY= dataY(A,:);
    prey=C4_5(subtrainx',subtrainY',validX',5);
    prey=prey';%һ�б�һ��
    va_prey=B_compute_label(prey,envelope);%������Ԥ���ǩת��ԭ����Ԥ���ǩ
    labelval=[labelval,va_prey];
    
    prey=C4_5(subtrainx',subtrainY',testX',5);
    prey=prey';%һ�б�һ��
    te_prey=B_compute_label(prey,envelope);%������Ԥ���ǩת��ԭ����Ԥ���ǩ
    labeltest=[labeltest,te_prey];    
end
end