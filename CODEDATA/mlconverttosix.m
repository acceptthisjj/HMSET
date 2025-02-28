function [U] = mlconverttosix(X,method,b) 
[m, n] = size(X);
U1 = zeros(6,n);
%第一次降维处理
U1(1,:)=mean(X,1);%求列均值
U1(2,:)=median(X,1);%中位数
U1(4,:)=std(X,1);% 计算标准差，权值为1，维度为1，也就是计算列标准差
U1(5,:)=mad(X);%平均绝对偏差
U1(6,:)=quantile(X,0.75,1)-quantile(X,0.25,1);%四分位范围
[val,ind]=sort(X);
U1(3,:)=mean(val(1+floor(b*0.2):1+floor(b*0.8),:),1);% 20%截尾均值 前1 后1
%第二次计算
U2 = zeros(6,n);

U2(1,:)=mean(U1,1);%求列均值
U2(2,:)=median(U1,1);%中位数
U2(4,:)=std(U1,1);% 计算标准差，权值为1，维度为1，也就是计算列标准差
U2(5,:)=mad(U1);%平均绝对偏差
U2(6,:)=quantile(U1,0.75,1)-quantile(U1,0.25,1);%四分位范围
[val2,ind2]=sort(U1);
U2(3,:)=mean(val2(2:5,:),1);% 20%截尾均值 前1 后1

switch method.mode
    case'141'
        U=cat(1,U1(1,:),U1(4,:));
    case'251'
        U=cat(1,U1(2,:),U1(5,:));
    case'361'
        U=cat(1,U1(3,:),U1(6,:));
    case'all1'
        U=U1;
    case'142'
        U=cat(1,U2(1,:),U2(4,:));
    case'252'
        U=cat(1,U2(2,:),U2(5,:));
    case'362'
        U=cat(1,U2(3,:),U2(6,:));
    case'all2'
        U=U2;
    case'1'
        U=U1(1,:);
    case'2'
        U=U1(2,:);
    case'3'
        U=U1(3,:); 
    case'4'
        U=U1(4,:);
    case'5'
        U=U1(5,:);
    case'6'
        U=U1(6,:);
end

end