function [trainX,trainY,testX,testY] = dividweightsample(X,Y,num,sample)
 %sample һ���˵�������
 %num ����ѵ��������
[m,n]=size(Y);
sub=m/sample;
randnum=randperm(sub);

trainX=[];
trainY=[];
testX=[];
testY=[];
for i=1:num
    key=randnum(i);
    trainX=[trainX;X(1+sample*(key-1):sample*key,:)];
    trainY=[trainY;Y(1+sample*(key-1):sample*key,:)];
end

for i=(num+1):sub
    key=randnum(i);
    testX=[testX;X(1+sample*(key-1):sample*key,:)];
    testY=[testY;Y(1+sample*(key-1):sample*key,:)];
end

end