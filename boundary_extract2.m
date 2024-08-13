%M-file ,boundary_extract.m   这个方法是最严谨的，目标点与邻近点很清楚的表达
%根据平面上投影坐标方位角判断边界点（这个效果是最严谨也是最好的） 邻近点number_of_neighbor数越大，max_diff对应的阈值也就设置稍微小点
%input_pnts(nx3)
function [boundary_pnts] = boundary_extract2(input_pnts,number_of_neighbor)  %（这个是理论最正确的，也是效果是最好的）
n=size(input_pnts,1);     %获得第一维度（行）的规模
k=number_of_neighbor;
neighbor_idx=knnsearch(input_pnts,input_pnts,'k',number_of_neighbor+1);  %含P点对应的k+1个邻近点在哪一行
normal_pnts=zeros(3,n);    %产生3*n的全零矩阵
line=zeros(n,1);
for i=1:n
    
  neighbor_pnts=input_pnts(neighbor_idx(i,:),:); 
  mean_neighbor=mean(neighbor_pnts,1);  
[parameter] = TLS_Plane(neighbor_pnts);
  normal_pnts =parameter(1:3,:);

 lamda=(repmat(mean_neighbor*normal_pnts,number_of_neighbor+1,1)-neighbor_pnts*normal_pnts)/(sum(normal_pnts.^2));% kx1列矩阵
 proj_pnts=normal_pnts*lamda'+neighbor_pnts';%3xk
 object_pnt=proj_pnts(:,1)';
 
 x_unit_vector=(proj_pnts(:,end)'-object_pnt)/norm(proj_pnts(:,end)'-object_pnt);% 通过探测点input_pnts(i,:)和投影邻近点最远的点proj_pnts(:,end)'构建x轴法向量 1x3 
 y_unit_vector=cross(normal_pnts,x_unit_vector)/norm(cross(normal_pnts,x_unit_vector)); %根据两个向量叉乘除以叉乘的模来确定另一个坐标轴上的单位向量 1x3
 
 x_plane=sum(repmat(x_unit_vector,k,1).*(proj_pnts(:,2:end)'-repmat(object_pnt,k,1)),2); % 根据单位向量乘以向量得到在平面x轴上的投影kx1
 y_plane=sum(repmat(y_unit_vector,k,1).*(proj_pnts(:,2:end)'-repmat(object_pnt,k,1)),2); % 根据单位向量乘以向量得到在平面y轴上的投影kx1
 
 azimuth= rad2deg(atan2(y_plane,x_plane)); %计算坐标方位角，如果是在第一和第二象限则是正值，如果是第三和第四象限则是负值
 azimuth(azimuth<0)= azimuth(azimuth<0)+360 ; %将坐标方位角的负值全部变为正值
sort_azimuth=sort(azimuth);
diff_azimuth=diff(sort_azimuth);
max_diff(i,:)=max(diff_azimuth);
end
idx=find(max_diff>120);
boundary_pnts=input_pnts(idx,:);