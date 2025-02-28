function [trainX,trainY,testX,testY] = dividSample_sub(X,Y,i,cc)
%
    trainX=[];
    trainY=[];
    
    testX = X(1+cc*(i-1):cc*i,:);
    testY = Y(1+cc*(i-1):cc*i,:);
    X(1+cc*(i-1):cc*i,:) = [];
    Y(1+cc*(i-1):cc*i,:) = [];
    
    %subject ´òÂÒ
    sub=size(Y,1)/cc;
    key=randperm(sub);
    
    for i=1:size(key,2)
    trainX=[trainX;X(1+cc*(key(i)-1):cc*key(i),:)];
    trainY=[trainY;Y(1+cc*(key(i)-1):cc*key(i),:)];
    end

end