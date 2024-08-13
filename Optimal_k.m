
%该命令是基于信息熵理论实现的最佳邻近点数判断准则
%参考文献：卢维欣,万幼川,何培培,等. 大场景内建筑物点云提取及平面分割算法[J]. 中国激光,2015,42(9):336-342. 
%输入数据：原始点云input_pnts（nx3），[min_k max_k] 最邻近点数的范围
%输出数据：每个点不同邻近点的信息熵E，每个点的最佳邻近点数best_k
function [E,best_k] = Optimal_k(input_pnts,min_k,max_k)


inter=max_k-min_k+1;
[neigh,~] = knnsearch(input_pnts,input_pnts, 'k', max_k);
for i=1:size(input_pnts,1)
for j=1:inter
    k=min_k+(j-1);
pnt=input_pnts(neigh(i,1:k),:);
n=size(pnt,1);
mean_pnts=mean(pnt,1);
M=pnt-repmat(mean_pnts,n,1);
[U,S,V] = svd(M'*M); %U 和V都是特征向量，S是特征值

a1=S(1,1);a2=S(2,2);a3=S(3,3);

a11=(sqrt(a1)-sqrt(a2))/sqrt(a1);
a12=(sqrt(a2)-sqrt(a3))/sqrt(a1);
a13=(sqrt(a3))/sqrt(a1);
E(j,i)=-a11*log(a11)-a12*log(a12)-a13*log(a13);
end
end
[min_E,index]=min(E,[],1);
best_k=min_k+(index-1);
