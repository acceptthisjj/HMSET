function [labelarr]=subspace2_C4_5(numtree,trainX,trainY,testX,envelope,class1,class2)
%��ö���ز����ӿռ�ѵ��������������ò��Ա�ǩ
%�������70%���ӿռ�
%���ӿռ�Ԥ����
labelarr=[];

[b1,m1,n1]=unique(trainX,'rows','stable'); %���Զ�����  ��������� ѡ�����������
%�޳��ظ�
dataX=b1;
dataY=trainY(m1,:);

m=size(dataY,1);
for i=1:numtree
    A=randperm(m,round(m*0.7));
    subtrainx= dataX(A,:);
    subtrainY= dataY(A,:);
%     a1=length(subtrainY==0)
%     b1=length(subtrainY==1)
    
    prey=C4_5(subtrainx',subtrainY',testX',5);
    prey=prey';%һ�б�һ��
    re_prey=compute_label(prey,envelope,class1,class2);%������Ԥ���ǩת��ԭ����Ԥ���ǩ
    labelarr=[labelarr,re_prey];
end
end

%�ز����ķ�ʽ C4.5���� ѵ���������ظ�������̫�ࣿ
% function [labelarr]=subspace_C4_5(numtree,trainX,trainY,testX,envelope,class1,class2)
% %��ö���ز����ӿռ�ѵ��������������ò��Ա�ǩ
% %�ز����ӿռ�
% [bootstat,bootsam] = bootstrp(numtree, @(x) mean(x), trainX);%�ز���
% %���ӿռ�Ԥ����
% labelarr=[];
% for i=1:size(bootsam,2)
%     subtrainx= trainX(bootsam(:,i),:);
%     subtrainY= trainY(bootsam(:,i),:);
%     prey=C4_5(subtrainx',subtrainY',testX',5);
%     prey=prey';%һ�б�һ��
%     re_prey=compute_label(prey,envelope,class1,class2);
%     labelarr=[labelarr,re_prey];
% end
% end
