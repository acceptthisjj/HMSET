%两类计算，其中一类标签值 正例标签（患者的类标签）
%
function [acc,sensitivity,specificity] = compute_ass(prelabel,tlabel,y) 
TP=0;
TN=0;
FN=0;
FP=0;
for i=1:size(prelabel,1)
if prelabel(i,1)==tlabel(i,1)
    if tlabel(i,1)==y%1-1
        TP=TP+1;
    else
        TN=TN+1;%0-0
    end
else
    if tlabel(i,1)==y%1-0
        FN=FN+1;
    else
        FP=FP+1;%0-1
    end
end
end
    sensitivity=TP/(TP+FN);
    specificity=TN/(FP+TN);
    acc=mean(prelabel==tlabel);
end