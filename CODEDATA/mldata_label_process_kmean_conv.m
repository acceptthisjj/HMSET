function [datanew,labelnew] = mldata_label_process_kmean_conv(train_data,train_label,sub,sam)
%数据处理
datanew=[];ans1=[];ans2=[];
labelnew=[];
for i=1:sub
x=train_data(1+sam*(i-1):sam+sam*(i-1),:);
y=train_label(1+sam*(i-1):sam+sam*(i-1),:);

[Idx,C,sumD,D]=kmeans(x,2,'dist','sqEuclidean','Start','sample');

table(i,1)=size(find(Idx==1),1);
table(i,2)=size(find(Idx==2),1);

c1=find(Idx==1);
c2=find(Idx==2);

[d1,l1,d2,l2]=perctsix(x,y,c1,c2);%聚类处理后 2*6  计算出单人聚类变换数据
r1=x(c1,:);%分类数据
r2=x(c2,:);

for num=1:size(d1,1)
    ans1=[ans1;sum(d1(num,:).*r1,1)]; %乘机求和 反卷
end
for num=1:size(d2,1)
    ans2=[ans2;sum(d2(num,:).*r2,1)]; %乘机求和 反卷
end
dataper=cat(1,ans1,ans2);%daigai
labelper=cat(1,l1,l2);
ans1=[];ans2=[];%清空

datanew=cat(1,datanew,dataper);
labelnew=cat(1,labelnew,labelper);
end
end