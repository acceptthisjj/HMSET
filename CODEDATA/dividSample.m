function [trainX,trainY,testX,testY] = dividSample(X,Y,i,cc)
    
%     ind = [];
%     group_pos = [1 23;24 45;46 68;69 91;92 114];
%     ind = group_pos(i,1):group_pos(i,2);
    testX = X(1+cc*(i-1):cc*i,:);
    testY = Y(1+cc*(i-1):cc*i,:);
    X(1+cc*(i-1):cc*i,:) = [];
    Y(1+cc*(i-1):cc*i,:) = [];
    train=[X';Y']';
    train=train(randperm(size(train,1)),:);%´òÂÒÅÅĞò
    trainX = train(:,1:end-1);
    trainY = train(:,end);
    

end