%按照key进行挑选  选择每个人中的部分片段
function [dataperonex,dataperoney] = data_label_select(data,label,num,key)
dataperonex=[];
dataperoney=[];
s_n=size(data,1)/num;
for i=1:num
    dataperx=data(1+s_n*(i-1):s_n+s_n*(i-1),:);%选择一个病人的片段
    datapery=label(1+s_n*(i-1):s_n+s_n*(i-1),:);
%     c=reshape(dataperx',1,[]);
    c=dataperx(key,:);
    dataperonex=[dataperonex;c];%将本裁剪
    dataperoney=[dataperoney;datapery(key,:)];
end
end