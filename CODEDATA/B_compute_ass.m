%������㣬��׼ȷ��
function [acc] = B_compute_ass(prelabel,tlabel) 
same=(prelabel==tlabel);
acc=mean(same);
end