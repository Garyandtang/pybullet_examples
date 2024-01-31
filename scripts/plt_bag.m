clear;
num_bags = 10;
vehicle_odoms = zeros(1000,2,num_bags);
vehicle_cmds = zeros(1000,2,num_bags);
bag_len = zeros(num_bags,1);
x_diff = zeros(num_bags,1);
y_diff = zeros(num_bags,1);
l2_diff = zeros(num_bags,1);
ang_diff = zeros(num_bags,1);
for i = 1:num_bags
%     if i ==7
%         continue
%     end
    if i<10
        bag_name=sprintf('%s%d%s','ps_1_5_0_6_1_0_0_5_0',i,'.bag');
%         bag_name=sprintf('%s%d%s','ps_1_5_0_6_1_0_1_0_0',i,'.bag');
    else
        bag_name=sprintf('%s%d%s','ps_1_5_0_6_1_0_0_5_',i,'.bag');
%         bag_name=sprintf('%s%d%s','ps_1_5_0_6_1_0_1_0_',i,'.bag');
    end

    bag = rosbag(bag_name);
    bag_len(i) = bag.NumMessages;
    current = select(bag,'Topic','/vehicle_pose','MessageType','nav_msgs/Odometry');
    vehicle_current = readMessages(current,'DataFormat','struct');
    for j=1:bag_len(i)
        vehicle_odoms(j,1,i) = vehicle_current{j,1}.Pose.Pose.Position.X;
        vehicle_odoms(j,2,i) = vehicle_current{j,1}.Pose.Pose.Position.Y;
    end
    w = vehicle_current{j,1}.Pose.Pose.Orientation.W;
    x = vehicle_current{j,1}.Pose.Pose.Orientation.X;
    y = vehicle_current{j,1}.Pose.Pose.Orientation.Y;
    z = vehicle_current{j,1}.Pose.Pose.Orientation.Z;
    eul = quat2eul([w x y z]);
    x_diff(i) = abs(3 - vehicle_odoms(bag_len(i),1,i));
    y_diff(i) = abs(0 - vehicle_odoms(bag_len(i),2,i));
    l2_diff(i) = sqrt(x_diff(i)^2+y_diff(i)^2);
    ang_diff(i) = abs(pi/6 - eul(1));
end

x_diff_mean = mean(x_diff);
y_diff_mean = mean(y_diff);
l2_diff_mean = mean(l2_diff);
x_diff_sd = std(x_diff);
y_diff_sd = std(y_diff);
l2_diff_sd = std(l2_diff);
x_diff = sort(x_diff);
y_diff = sort(y_diff);
l2_diff = sort(l2_diff);
ang_diff = sort(ang_diff);
boxplot([x_diff(2:end-1),y_diff(2:end-1),l2_diff(2:end-1),ang_diff(2:end-1)],'Labels',{'x_diff','y_diff','l2_diff','angle_dlff'},'Whisker',10);
title('Boxplot of the errors')
ylabel('Errors (m or rad)');


num_forward=8;
path_type_forward=[0;1;0;1;0;1;0;1];% 1 stands for bezier curve, 0 stands for straigt line
P_forward=zeros(2,num_forward*4);

P_forward(1:2,1:4)=[0,0,0,4;2,0,0,2];
P_forward(1:2,5:8)=[4,4.5477,5,5;2,2,1.5477,1];
P_forward(1:2,9:12)=[5,0,0,5;1,0,0,-2];
P_forward(1:2,13:16)=[5,5,4.5477,4;-2,-2.5477,-3,-3];
P_forward(1:2,17:20)=[4,0,0,1;-3,0,0,-3];
P_forward(1:2,21:24)=[1,0.4532,0,0;-3,-3,-2.5477,-2];
P_forward(1:2,25:28)=[0,0,0,0;-2,0,0,-1];
P_forward(1:2,25:28)=[0,0,0,0;-2,0,0,-1];
P_forward(1:2,29:32)=[0,0,2.0341,3;-1,0.6201,-0.5577,0];

dt=0.005;
step_total=floor(8/dt);
step=floor(step_total/8);
x_ori=zeros(8*step+1,1);
y_ori=zeros(8*step+1,1);
dk=1/step;
x_ori(1) = P_forward(1,1);
y_ori(1) = P_forward(2,1);
for k=1:8
    if path_type_forward(k)==1
        for i=1:step %calculate the points on original bezier curve
            [x_ori(1+i+(k-1)*step),y_ori(1+i+(k-1)*step)]=calculate_bezier_point(P_forward(1:2,(k-1)*4+1:4*k),i*dk);
        end
    else
        x0=P_forward(1,(k-1)*4+1);
        y0=P_forward(2,(k-1)*4+1);
        x_end=P_forward(1,4*k);
        y_end=P_forward(2,4*k);
        Len=norm([y_end-y0;x_end-x0]);
        theta_line=atan2(y_end-y0,x_end-x0);
        for i=1:step
            x_ori(1+i+(k-1)*step)=x0+i*dk*Len*cos(theta_line);
            y_ori(1+i+(k-1)*step)=y0+i*dk*Len*sin(theta_line);
        end
    end
end
color = zeros(10,3);
color(1,:) = [255 5 5]/255;
color(2,:) = [255 100 5]/255;
color(3,:) = [255 255 5]/255;
color(4,:) = [200 255 5]/255;
color(5,:) = [5 255 5]/255;
color(6,:) = [5 255 100]/255;
color(7,:) = [5 255 255]/255;
color(8,:) = [5 200 255]/255;
color(9,:) = [5 5 255]/255;
color(10,:) = [255 5 255]/255;

figure;
hold on;
% t_xer1 = ['average x error: ', num2str(mean(x_diff(2:end-1)))];
% t_yer1 = ['average y error: ', num2str(mean(y_diff(2:end-1)))];
% t_l2er1 = ['average l2 error: ', num2str(mean(l2_diff(2:end-1)))];
% t_anger1 = ['average ang error: ', num2str(mean(ang_diff(2:end-1)))];
% title({'qv 0.5';t_xer1;t_yer1;t_l2er1;t_anger1})
plot(x_ori,y_ori,'-k','LineWidth',3);
for i=1:1
    plot(vehicle_odoms(1:bag_len(i),1,i),vehicle_odoms(1:bag_len(i),2,i),'-','LineWidth',1,'Color',color(i,:));
end
hold off;
xlabel('m');
ylabel('m');
xlim([-1 6]);
ylim([-3.5 2.5]);
grid on;
axis equal;