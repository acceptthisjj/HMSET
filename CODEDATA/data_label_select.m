%����key������ѡ  ѡ��ÿ�����еĲ���Ƭ��
function [dataperonex,dataperoney] = data_label_select(data,label,num,key)
dataperonex=[];
dataperoney=[];
s_n=size(data,1)/num;
for i=1:num
    dataperx=data(1+s_n*(i-1):s_n+s_n*(i-1),:);%ѡ��һ�����˵�Ƭ��
    datapery=label(1+s_n*(i-1):s_n+s_n*(i-1),:);
%     c=reshape(dataperx',1,[]);
    c=dataperx(key,:);
    dataperonex=[dataperonex;c];%�����ü�
    dataperoney=[dataperoney;datapery(key,:)];
end
end