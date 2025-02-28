function [datanew,labelnew] = mldata_label_process_kmean(train_data,train_label,sub,sam)
%数据处理
%sub 包络数
%sam 每个包络的样本数
datanew=[];
labelnew=[];
for i=1:sub
x=train_data(1+sam*(i-1):sam+sam*(i-1),:);
y=train_label(1+sam*(i-1):sam+sam*(i-1),:);

[Idx,C,sumD,D]=kmeans(x,2,'dist','sqEuclidean','Start','sample');

table(i,1)=size(find(Idx==1),1);
table(i,2)=size(find(Idx==2),1);

c1=find(Idx==1);
c2=find(Idx==2);

[d1,l1,d2,l2]=perctsix(x,y,c1,c2);%聚类处理后 2*6

dataper=cat(1,d1,d2);
labelper=cat(1,l1,l2);
datanew=cat(1,datanew,dataper);
labelnew=cat(1,labelnew,labelper);
end
end