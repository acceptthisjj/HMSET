
close all;
clear;
clc;

load('./ExperimentData/wine')% 读取数据集，根据数据集修改

data_all=data;
label_all=label;

rng(1);
Ptest=0.2;%6:2:2
Pvalid=0.25;
%
p1_num=6;p2_num=12;p3_num=12;
% weight=0.5;%近邻样本权重
kk=1;%特征间隔
%
xinta=0.01;
all_predict_label=[];
perweight=0.45;
mainweight=1;
%% 验证集搜索最优参数
for ik=1:floor(size(data_all,2)/kk)%
method.K = kk*ik;
%数据集 验证集 测试集
[idxtra,idxtest] = crossvalind('HoldOut', size(data_all,1),Ptest);
tra1X=data_all(idxtra,:);tra1Y=label_all(idxtra,:);
testX=data_all(idxtest,:);testY=label_all(idxtest,:);
[idxtrain,idxvalid] = crossvalind('HoldOut', size(tra1X,1),Pvalid);
trainX=tra1X(idxtrain,:);trainY=tra1Y(idxtrain,:);
validX=tra1X(idxvalid,:);validY=tra1Y(idxvalid,:);
%标准化
[trainX, mu, sigma] = featureCentralize(trainX);
testX = bsxfun(@minus, testX, mu);
testX = bsxfun(@rdivide, testX, sigma);
validX = bsxfun(@minus, validX, mu);
validX = bsxfun(@rdivide, validX, sigma);
%k近邻 %权重
% knn_k=ceil(size(trainX,1)*0.02);%自适应包络样本数
% if knn_k>9
%     knn_k=9;
% end
knn_k=9;

[trainX_kn,trainY_kn]=B_KN_cedi_train_creat(trainX,trainY,knn_k,mainweight,perweight*knn_k);
[validX_kn,validY_kn]=B_KN_cedi_test_creat(trainX,trainY,validX,validY,knn_k,mainweight,perweight*knn_k);
[testX_kn,testY_kn]=B_KN_cedi_test_creat(trainX,trainY,testX,testY,knn_k,mainweight,perweight*knn_k);
%样本变换
    [trainX_p1,trainY_p1]=process1(trainX_kn,trainY_kn,4,size(trainX,1),knn_k+1);
    [testX_p1,testY_p1]=process1(testX_kn,testY_kn,4,size(testX,1),knn_k+1);
    [validX_p1,validY_p1]=process1(validX_kn,validY_kn,4,size(validX,1),knn_k+1);
    [trainX_p2,trainY_p2]=mldata_label_process_kmean(trainX_kn,trainY_kn,size(trainX,1),knn_k+1);
    [testX_p2,testY_p2]=mldata_label_process_kmean(testX_kn,testY_kn,size(testX,1),knn_k+1);
    [validX_p2,validY_p2]=mldata_label_process_kmean(validX_kn,validY_kn,size(validX,1),knn_k+1);
    [trainX_p3,trainY_p3]=mldata_label_process_kmean_conv(trainX_kn,trainY_kn,size(trainX,1),knn_k+1);
    [testX_p3,testY_p3]=mldata_label_process_kmean_conv(testX_kn,testY_kn,size(testX,1),knn_k+1);
    [validX_p3,validY_p3]=mldata_label_process_kmean_conv(validX_kn,validY_kn,size(validX,1),knn_k+1);
%防止0值样本
b=10^-4;
trainX_p1=fitzero(trainX_p1,b);testX_p1=fitzero(testX_p1,b);validX_p1=fitzero(validX_p1,b);
trainX_p2=fitzero(trainX_p2,b);testX_p2=fitzero(testX_p2,b);validX_p2=fitzero(validX_p2,b);
trainX_p3=fitzero(trainX_p3,b);testX_p3=fitzero(testX_p3,b);validX_p3=fitzero(validX_p3,b);
%域适应
    %参数设置
    gamma=100;
    Init_options.lambda=0.1;  %  
    Init_options.dim=size(data_all,2);  %JPSCA  域适应维度跟随变换------------------------
    Init_options.kernel_type='primal';  %不用核函数  
    Init_options.gamma=gamma;  %  rbf  
    Init_options.T=1;
    Init_options.weightmode='binary';
    Init_options.mode='lpp';

    %域适应-p1
    data_src=trainX;
    data_tar=trainX_p1;
    [trainX_S_p1,trainX_J_p1,A] = JPSCA(data_src,data_tar,Init_options);% 
    %根据系数调整矩阵
    A_Z=A;
    trainX_J_p1=projectData_center(data_tar,A_Z,Init_options.dim);%
    validX_J_p1=projectData_center(validX_p1,A_Z,Init_options.dim);%
    testX_J_p1=projectData_center(testX_p1,A_Z,Init_options.dim);% 
    %域适应-p2
    data_src=trainX;
    data_tar=trainX_p2;
    [trainX_S_p2,trainX_J_p2,A] = JPSCA(data_src,data_tar,Init_options);% 
    A_Z=A;
    trainX_J_p2=projectData_center(data_tar,A_Z,Init_options.dim);%    
    validX_J_p2=projectData_center(validX_p2,A_Z,Init_options.dim);%
    testX_J_p2=projectData_center(testX_p2,A_Z,Init_options.dim);%    
    %域适应-p3
    data_src=trainX;
    data_tar=trainX_p3;
    [trainX_S_p3,trainX_J_p3,A] = JPSCA(data_src,data_tar,Init_options);% 
    A_Z=A;
    trainX_J_p3=projectData_center(data_tar,A_Z,Init_options.dim);%        
    validX_J_p3=projectData_center(validX_p3,A,Init_options.dim);%
    testX_J_p3=projectData_center(testX_p3,A,Init_options.dim);% 
%% pca
    method.mode = 'pca';
    type_num=2;
    %原数据r-pca 子空间4
    U= featureExtract(trainX,trainY,method,type_num);
    trainX_pca_p4=projectData(trainX, U, method.K);%
    testX_pca_p4=projectData(testX, U, method.K);%
    validX_pca_p4=projectData(validX, U, method.K);%
    %参数寻优中可适当减少计算量
    [label_r_pca]=B_subspace2_C4_5(10,trainX_pca_p4,trainY,validX_pca_p4,1);  
    
    %p1-迁-pca 子空间1
    U= featureExtract(trainX_J_p1,trainY_p1,method,type_num);
    trainX_J_pca_p1=projectData(trainX_J_p1, U, method.K);%
    testX_J_pca_p1=projectData(testX_J_p1, U, method.K);%
    validX_J_pca_p1=projectData(validX_J_p1, U, method.K);%
    %
    [label_p1_pca]=B_subspace2_C4_5(10,trainX_J_pca_p1,trainY_p1,validX_J_pca_p1,p1_num);
    
    %p2-迁-pca 子空间2
    U= featureExtract(trainX_J_p2,trainY_p2,method,type_num);
    trainX_J_pca_p2=projectData(trainX_J_p2, U, method.K);%
    testX_J_pca_p2=projectData(testX_J_p2, U, method.K);%
    validX_J_pca_p2=projectData(validX_J_p2, U, method.K);%

    [label_p2_pca]=B_subspace2_C4_5(10,trainX_J_pca_p2,trainY_p2,validX_J_pca_p2,p2_num);
    
    %p3-迁-pca 子空间3
    U= featureExtract(trainX_J_p3,trainY_p3,method,type_num);
    trainX_J_pca_p3=projectData(trainX_J_p3, U, method.K);%
    testX_J_pca_p3=projectData(testX_J_p3, U, method.K);%
    validX_J_pca_p3=projectData(validX_J_p3, U, method.K);%
    
    [label_p3_pca]=B_subspace2_C4_5(10,trainX_J_pca_p3,trainY_p3,validX_J_pca_p3,p3_num);
    
    %拿到验证集结果
    %统合结果 这里存储了不同维度下的验证集结果
    all_predict_label=[label_r_pca,label_p1_pca,label_p2_pca,label_p3_pca];
    [acc1_Jpca(ik)]=B_compute_ass(mode(label_p1_pca,2),validY);%    
    [acc2_Jpca(ik)]=B_compute_ass(mode(label_p2_pca,2),validY);%
    [acc3_Jpca(ik)]=B_compute_ass(mode(label_p3_pca,2),validY);%
    [acc_rpca(ik)]=B_compute_ass(mode(label_r_pca,2),validY);%
    [all_acc_pca(ik)]=B_compute_ass(mode(all_predict_label,2),validY);%
    
    value_acc=all_acc_pca;%用于找最优降维
    %
    ikc=ik
end

save C_test_weight_wine value_acc kk

% %主要方法代码 整体流程 
close all;
clear;
clc;

load C_test_weight_wine% 读取最优参数，根据数据集修改

[~,max_idx]=max(value_acc);

load('./ExperimentData/wine')
data_all=data;
label_all=label;
rng(1);
Ptest=0.2;%6:2:2
Pvalid=0.25;
p1_num=6;p2_num=12;p3_num=12;
xinta=0.01;
all_predict_label=[];
perweight=0.45;
mainweight=1;

method.K = kk*max_idx;%------------------------------

for ik=1:10
% % 数据集 验证集 测试集
[idxtra,idxtest] = crossvalind('HoldOut', size(data_all,1),Ptest);
tra1X=data_all(idxtra,:);tra1Y=label_all(idxtra,:);
testX=data_all(idxtest,:);testY=label_all(idxtest,:);
[idxtrain,idxvalid] = crossvalind('HoldOut', size(tra1X,1),Pvalid);
trainX=tra1X(idxtrain,:);trainY=tra1Y(idxtrain,:);
validX=tra1X(idxvalid,:);validY=tra1Y(idxvalid,:);
% % 标准化
[trainX, mu, sigma] = featureCentralize(trainX);
testX = bsxfun(@minus, testX, mu);
testX = bsxfun(@rdivide, testX, sigma);
validX = bsxfun(@minus, validX, mu);
validX = bsxfun(@rdivide, validX, sigma);
% % k近邻 %权重
% knn_k=ceil(size(trainX,1)*0.02);%自适应包络样本数
% if knn_k>9
%     knn_k=9;
% end
knn_k=9;

[trainX_kn,trainY_kn]=B_KN_cedi_train_creat(trainX,trainY,knn_k,mainweight,perweight*knn_k);
[validX_kn,validY_kn]=B_KN_cedi_test_creat(trainX,trainY,validX,validY,knn_k,mainweight,perweight*knn_k);
[testX_kn,testY_kn]=B_KN_cedi_test_creat(trainX,trainY,testX,testY,knn_k,mainweight,perweight*knn_k);
% % 样本变换
    [trainX_p1,trainY_p1]=process1(trainX_kn,trainY_kn,4,size(trainX,1),knn_k+1);
    [testX_p1,testY_p1]=process1(testX_kn,testY_kn,4,size(testX,1),knn_k+1);
    [validX_p1,validY_p1]=process1(validX_kn,validY_kn,4,size(validX,1),knn_k+1);
    [trainX_p2,trainY_p2]=mldata_label_process_kmean(trainX_kn,trainY_kn,size(trainX,1),knn_k+1);
    [testX_p2,testY_p2]=mldata_label_process_kmean(testX_kn,testY_kn,size(testX,1),knn_k+1);
    [validX_p2,validY_p2]=mldata_label_process_kmean(validX_kn,validY_kn,size(validX,1),knn_k+1);
    [trainX_p3,trainY_p3]=mldata_label_process_kmean_conv(trainX_kn,trainY_kn,size(trainX,1),knn_k+1);
    [testX_p3,testY_p3]=mldata_label_process_kmean_conv(testX_kn,testY_kn,size(testX,1),knn_k+1);
    [validX_p3,validY_p3]=mldata_label_process_kmean_conv(validX_kn,validY_kn,size(validX,1),knn_k+1);
% % 防止0值样本
b=10^-4;
trainX_p1=fitzero(trainX_p1,b);testX_p1=fitzero(testX_p1,b);validX_p1=fitzero(validX_p1,b);
trainX_p2=fitzero(trainX_p2,b);testX_p2=fitzero(testX_p2,b);validX_p2=fitzero(validX_p2,b);
trainX_p3=fitzero(trainX_p3,b);testX_p3=fitzero(testX_p3,b);validX_p3=fitzero(validX_p3,b);
% % 域适应
% %     参数设置
    gamma=100;
    Init_options.lambda=0.1;  %  
    Init_options.dim=size(data_all,2);  %JPSCA  域适应维度跟随变换------------------------
    Init_options.kernel_type='primal';  
    Init_options.gamma=gamma;  %  rbf  
    Init_options.T=1;
    Init_options.weightmode='binary';
    Init_options.mode='lpp';
% %    域适应-p1
    data_src=trainX;
    data_tar=trainX_p1;
    [trainX_S_p1,trainX_J_p1,A] = JPSCA(data_src,data_tar,Init_options);% 
% %     根据系数调整矩阵
    A_Z=A;
    trainX_J_p1=projectData_center(data_tar,A_Z,Init_options.dim);%
    validX_J_p1=projectData_center(validX_p1,A_Z,Init_options.dim);%
    testX_J_p1=projectData_center(testX_p1,A_Z,Init_options.dim);% 
% %     域适应-p2
    data_src=trainX;
    data_tar=trainX_p2;
    [trainX_S_p2,trainX_J_p2,A] = JPSCA(data_src,data_tar,Init_options);% 
    A_Z=A;
    trainX_J_p2=projectData_center(data_tar,A_Z,Init_options.dim);%    
    validX_J_p2=projectData_center(validX_p2,A_Z,Init_options.dim);%
    testX_J_p2=projectData_center(testX_p2,A_Z,Init_options.dim);%    
% %     域适应-p3
    data_src=trainX;
    data_tar=trainX_p3;
    [trainX_S_p3,trainX_J_p3,A] = JPSCA(data_src,data_tar,Init_options);% 
    A_Z=A;
    trainX_J_p3=projectData_center(data_tar,A_Z,Init_options.dim);%        
    validX_J_p3=projectData_center(validX_p3,A,Init_options.dim);%
    testX_J_p3=projectData_center(testX_p3,A,Init_options.dim);% 
% pca
    method.mode = 'pca';
    type_num=2;
    
% %     原数据r-pca 子空间4
    U= featureExtract(trainX,trainY,method,type_num);
    trainX_pca_p4=projectData(trainX, U, method.K);%
    testX_pca_p4=projectData(testX, U, method.K);%
    validX_pca_p4=projectData(validX, U, method.K);%
    
    [label_r_pca]=B_subspace2_C4_5(10,trainX_pca_p4,trainY,testX_pca_p4,1);  
    
% %     p1-迁-pca 子空间1
    U= featureExtract(trainX_J_p1,trainY_p1,method,type_num);
    trainX_J_pca_p1=projectData(trainX_J_p1, U, method.K);%
    testX_J_pca_p1=projectData(testX_J_p1, U, method.K);%
    validX_J_pca_p1=projectData(validX_J_p1, U, method.K);%
    
    [label_p1_pca]=B_subspace2_C4_5(10,trainX_J_pca_p1,trainY_p1,testX_J_pca_p1,p1_num);
    
% %     p2-迁-pca 子空间2
    U= featureExtract(trainX_J_p2,trainY_p2,method,type_num);
    trainX_J_pca_p2=projectData(trainX_J_p2, U, method.K);%
    testX_J_pca_p2=projectData(testX_J_p2, U, method.K);%
    validX_J_pca_p2=projectData(validX_J_p2, U, method.K);%

    [label_p2_pca]=B_subspace2_C4_5(10,trainX_J_pca_p2,trainY_p2,testX_J_pca_p2,p2_num);
    
% %     p3-迁-pca 子空间3
    U= featureExtract(trainX_J_p3,trainY_p3,method,type_num);
    trainX_J_pca_p3=projectData(trainX_J_p3, U, method.K);%
    testX_J_pca_p3=projectData(testX_J_p3, U, method.K);%
    validX_J_pca_p3=projectData(validX_J_p3, U, method.K);%
    
    [label_p3_pca]=B_subspace2_C4_5(10,trainX_J_pca_p3,trainY_p3,testX_J_pca_p3,p3_num);

% %     统合测试集结果
    all_predict_label=[label_r_pca,label_p1_pca,label_p2_pca,label_p3_pca];
% %     每个空间的预测标签
    l1=mode(label_p1_pca,2);
    l2=mode(label_p2_pca,2);
    l3=mode(label_p3_pca,2);
    lr=mode(label_r_pca,2);
    
    [acc1_Jpca(ik)]=B_compute_ass(l1,testY);%    
    [acc2_Jpca(ik)]=B_compute_ass(l2,testY);%
    [acc3_Jpca(ik)]=B_compute_ass(l3,testY);%
    [acc_rpca(ik)]=B_compute_ass(lr,testY);%
    [all_acc_pca(ik)]=B_compute_ass(mode(all_predict_label,2),testY);%投票集合

    ik
end
fprintf('mean acc：%.4f\n', mean(all_acc_pca));
fprintf('std acc：%.4f\n', std(all_acc_pca));