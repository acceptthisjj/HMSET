function [all_data,all_label]=splitbaseclass(trainX,trainY,percent)
class=unique(trainY);
data=[];
label=[];
for i=1:size(class,1)%����ÿһ��
    idx=find(trainY==class(i,1));%λ��
    m=size(idx,1);%����
    A=randperm(m,round(m*percent));
    pertrainX=trainX(idx(A,1),:);
    pertrainY=trainY(idx(A,1),:);
    data=[data;pertrainX];
    label=[label;pertrainY];
end
%����
f=size(label,1);
A1=randperm(f,f);
all_data=data(A1,:);
all_label=label(A1,:);
end