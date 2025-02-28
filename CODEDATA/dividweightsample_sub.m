function [trainX,trainY,testX,testY] = dividweightsample_sub(X,Y,prop,sample)
 %sample 一个人的样本数
 %prop 划分训比例 1:3 1train 3test
 trainX=[];
trainY=[];
testX=[];
testY=[];
[m,n]=size(Y);
sub=m/sample;
for i=1:sub
    sub_label(i)=Y(1+sample*(i-1),1);
end
per1=find(sub_label==1);
per0=find(sub_label==0);

p1=size(per1,2);
p0=size(per0,2);
per1=per1(1,randperm(p1));%打乱
per0=per0(1,randperm(p0));
trainkey=[per1(1,1:ceil(prop*p1)) per0(1,1:ceil(prop*p0))];
testkey=[per1(1,ceil(prop*p1)+1:end) per0(1,ceil(prop*p0)+1:end)];

for i=1:size(trainkey,2)
    key=trainkey(1,i);
    trainX=[trainX;X(1+sample*(key-1):sample*key,:)];
    trainY=[trainY;Y(1+sample*(key-1):sample*key,:)];
end

for i=1:size(testkey,2)
    key=testkey(1,i);
    testX=[testX;X(1+sample*(key-1):sample*key,:)];
    testY=[testY;Y(1+sample*(key-1):sample*key,:)];
end

end