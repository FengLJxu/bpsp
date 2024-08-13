%在已知几个三维坐标点的情况下进行平面方程的解算（z=ax+by+c）
%采用的是正交总体最小二乘的形式：a(x-xj)+b(y-yj)+c(z-zj)=0  ax+by+cz+d=0 （a,b,c）向量为最小特征值所对应的特征向量
%输入变量:input_pnts(nx3)   

function [parameter] = TLS_Plane(input_pnts)
n=size(input_pnts,1);
mean_pnts=mean(input_pnts,1);
M=input_pnts-repmat(mean_pnts,n,1);
[U,S,V] = svd(M'*M);
a=U(1,3);
b=U(2,3);
c=U(3,3);
d=-mean_pnts*[a;b;c];
parameter=[a;b;c;d];