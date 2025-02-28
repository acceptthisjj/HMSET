%多类计算，出准确率
function [acc] = B_compute_ass(prelabel,tlabel) 
same=(prelabel==tlabel);
acc=mean(same);
end