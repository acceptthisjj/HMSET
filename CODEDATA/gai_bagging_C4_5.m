function [labelval,labeltest]=gai_bagging_C4_5(numtree,trainX,trainY,validX,testX,percent)
%��������ȡѵ�������������
%��ȡ��������������Ԥ���ǩ
labelval=[];
labeltest=[];

% [b1,m1,n1]=unique(trainX,'rows','stable'); %���Զ�����  ��������� ѡ�����������
% %�޳��ظ�
% dataX=b1;
% dataY=trainY(m1,:);
% m=size(dataY,1);

for i=1:numtree
%     A=randperm(m,round(m*percent));%����飿������С����ʱ���·���ʧ��
%     subtrainx= dataX(A,:);
%     subtrainY= dataY(A,:);
    [subtrainx,subtrainY]=splitbaseclass(trainX,trainY,percent);%����������
    va_prey=Use_C4_5(subtrainx',subtrainY',validX',5,10);
    labelval=[labelval,va_prey'];
    te_prey=Use_C4_5(subtrainx',subtrainY',testX',5,10);
    labeltest=[labeltest,te_prey'];    
end

end