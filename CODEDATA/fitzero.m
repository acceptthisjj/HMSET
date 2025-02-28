function [data]=fitzero(data,plus)
[m,n]=find(data==0);
data(m,n)=data(m,n)+plus;
end