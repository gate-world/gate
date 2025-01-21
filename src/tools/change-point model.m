clc;
clear all;
%% 加载数据
ca1_response = load('ca1_response.mat').tensor;
ca1_response_dorsal = squeeze(ca1_response(1:640, 1, :, :));

%% 绘制热力图，找出明显有switch现象的神经元
% for cell=1:100
%     figure;
%     ca1_fr = squeeze(ca1_response_dorsal(:, :, cell));
%     h = heatmap(ca1_fr);
%     h.XLabel = 'position';
%     h.YLabel = 'trials';
%     colormap('jet');
%     h.Colormap = colormap;
%     h.GridVisible = 'off';
% end

%% 切出单个神经元不同trial的发放率矩阵，得到发放率峰值随trial的分布情况
firing_field = squeeze(ca1_response_dorsal(:, :, 49));
peak_fr = max(firing_field, [], 2);
% 0个change point的基准cost
c1 = sum((peak_fr-mean(peak_fr)).^2);

% 1个change point
c2 = c1;
T1 = 0; % 初始化切分点
for t=2:639
    cost1 = cost(peak_fr(1:t))+cost(peak_fr(t+1:640)); % 切分后两端序列分别计算cost
    if cost1 < c2
        c2 = cost1; % 但cost减小时，更新切分点
        T1 = t;
    end
end

% 2个change point
c3 = c2;
T2 = 0;
for t=2:639
    lower_bound = min(t, T1); % 固定T1，寻找使得cost最小的第二个切分点
    upper_bound = max(t, T1);
    part1 = peak_fr(1: lower_bound); % 根据切分点对序列进行切分
    part2 = peak_fr(lower_bound+1: upper_bound);
    part3 = peak_fr(upper_bound+1: 640);
    cost2 = cost(part1)+cost(part2)+cost(part3);
    if cost2<c3
        c3 = cost2;
        T2 = t;
    end
end

%% 图像
figure;
hold on;
plot(peak_fr, 'LineWidth', 2, 'Color', [0.5,0.5,0.5],'LineStyle','--'); % 灰色虚线表示真实peak fr
ylabel('peak firing rate');
xlabel('trials');
grid off;
state1 = mean(peak_fr(1: min(T1, T2)));
state2 = mean(peak_fr(min(T1, T2)+1: max(T1, T2)));
state3 = mean(peak_fr(max(T1, T2)+1: 640));
sequence1 = repmat(state1, 1, min(T1, T2));
sequence2 = repmat(state2, 1, max(T1, T2)-min(T1, T2));
sequence3 = repmat(state3, 1, 640-max(T1, T2));
x = 1:640;
y = [sequence1, sequence2, sequence3];
plot(x, y, 'b-', 'LineWidth', 2);

%计算R2
ss_tot = sum((peak_fr-mean(peak_fr)).^2);
ss_res = sum((peak_fr'-y).^2);
CPMr2 = 1-(ss_res/ss_tot); % change point model的R2
p = polyfit(x, peak_fr, 2); % 多项式的次数等于切换点的个数
y_fit = polyval(p, x);
plot(x, y_fit, 'r--', 'LineWidth', 2);
regr2 = 1-(sum((peak_fr'-y_fit).^2)/ss_tot); % 多项式回归的R2

function result = cost(sequence)
    mean_value = mean(sequence);
    result = sum((sequence-mean_value).^2);
end