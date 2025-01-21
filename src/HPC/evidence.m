clc;
clear all;
%% 加载数据
ca1_response = load('ca1_response.mat'); % ca1表征 epoch*tracklength*bs*ca1num
ca1_response = ca1_response.tensor;

trial_target = load('trial_target.mat'); % 奖励存在的位置 epoch*bs
trial_target = trial_target.tensor;

cue_train = load('trial_cue_train.mat'); % 探索过程中线索列 epoch*bs*tracklength
cue_train = cue_train.tensor;

%% 数据切片:选取最后一个epoch的全体ca1数据
% generete evidence train
bs = 1000;
epochnum = 250;
cue_train = squeeze(cue_train(epochnum, :, :));
target = squeeze(trial_target(epochnum, :));
evidence_train = zeros(size(cue_train));
for i = 1:100
    evidence_train(:, i) = sum(cue_train(:, 1:i), 2); % bs*position
end
max_evidence = max(evidence_train(:));
min_evidence = min(evidence_train(:));

ca1_fr = squeeze(ca1_response( :, :, :)); % tracklength*bs*ca1num
minValue = min(ca1_fr(:)); % max firing rate and min firing rate
maxValue = max(ca1_fr(:));

% evidence-position plat
% for cell = 1:100
%     figure;
%     init_space = zeros(bs, 100, max_evidence-min_evidence+1); % 创建一个空的二维空间
%     for trial = 1:bs
%         for x = 1:100
%             init_space(trial, x, evidence_train(trial, x)-min_evidence+1) = ca1_fr(x, trial, cell);
%         end
%     end
%     meanValue = zeros(100, size(init_space, 3));
%     stdValue = zeros(100, size(init_space, 3));
%     for j = 1:100
%         for k = 1:size(init_space, 3)
%             currentPlane = init_space(:, j, k);
%             meanValue(j, k) = mean(currentPlane(currentPlane ~= 0));
%             stdValue(j, k) = std(currentPlane(currentPlane ~= 0));
%         end
%     end
%     h = heatmap(meanValue);
%     h.XLabel = 'evidence';
%     h.XDisplayLabels = {'-15','-14','-13','-12','-11','-10','-9','-8','-7','-6','-5','-4','-3','-2','-1','0','1','2','3','4','5','6','7','8','9','10','11','12','13','14','15'};
%     h.YLabel = 'position';
%     h.YDisplayLabels = {'1','','','','','','','','','','','','','','','','','','','','','','','','','','','','','','','','','','','','','','','','','','','','','','','','','50','','','','','','','','','','','','','','','','','','','','','','','','','','','','','','','','','','','','','','','','','','','','','','','','','','100'};
%     h.GridVisible = 'off';
%     h.Title = sprintf('firing rate of CA1# %d', cell);
%     colormap('jet');
%     h.Colormap = colormap;
%     h.ColorLimits = [minValue, maxValue];
%     filename = sprintf('firing_rate %d.png', cell);
%     exportgraphics(h.Parent, filename, 'Resolution', 300);
%     close(h.Parent);
% end
% 
% figure;
% for p =1:bs
%     plot(evidence_train(p, :), 1:100, 'Color', [0.5+0.0003*bs,0.5+0.0003*bs,0.5+0.0003*bs], 'LineWidth', 2);
%     hold on;
% end
% xlabel('Evidence');
% ylabel('Position');
% title('Evidence-Position in 2D space');
% ax = gca; % 获取当前坐标轴
% ax.YTick = [1 50 100];
% ax.YTickLabel = {'1','50','100'};
% ax.XColor = 'k'; % 设置 X 轴颜色为黑色
% ax.YColor = 'k'; % 设置 Y 轴颜色为黑色
% ax.LineWidth = 1; % 设置坐标轴线的宽度

%% 看看某个trial的ca1表征
trial = 7;
target = target(trial); % 1表示左端有奖励，0表示右端有奖励
cell_preference = [1,0,1,0,0,0,0,0,1,0,1,0,1,0,1,1,1,1,1,0,0,0,1,1,0,1,0,1,0,1,0,0,1,0,0,0,1,0,0,0,0,0,0,0,1,0,1,0,0,1,1,0,1,0,1,0,1,1,1,0,0,1,0,0,0,1,1,1,1,1,0,0,1,0,0,0,0,1,0,0,1,0,1,0,0,1,0,1,1,1,0,1,1,0,1,0,0,0,1,0];
ca1_fr = squeeze(ca1_response(:, trial, :)); % tracklength*ca1num
cue = squeeze(cue_train(trial, :)); % tracklength
ymax = max(ca1_fr(:));

figure('Position', [100, 100, 1000, 600]);
hold on;
% 设置图例
h1 = plot(nan, nan, 's', 'Color', [1, 0.5, 0.5], 'MarkerFaceColor', [1, 0.5, 0.5], 'MarkerSize', 10);
h2 = plot(nan, nan, 's', 'Color', [0.5, 0.5, 1], 'MarkerFaceColor', [0.5, 0.5, 1], 'MarkerSize', 10);
h3 = plot(nan, nan, '--', 'Color', [1,0.7,0.7], 'LineWidth', 2);
h4 = plot(nan, nan, '--', 'Color', [0.7,0.7,1], 'LineWidth', 2);
for x = 1:100
    if cue(x) == 1
        rectangle('Position', [x-1, 0, 1, ymax], 'EdgeColor', [1, 0.5, 0.5], 'FaceColor', [1, 0.5, 0.5]);
    elseif cue(x) == -1
        rectangle('Position', [x-1, 0, 1, ymax], 'EdgeColor', [0.5, 0.5, 1], 'FaceColor', [0.5, 0.5, 1]);
    end
end
for cell =1:100
    if cell_preference(cell) == 1
        plot(1:100, ca1_fr(:, cell), '--', 'Color', [1,0.7,0.7], 'LineWidth', 2);
    else
        plot(1:100, ca1_fr(:, cell), '--', 'Color', [0.7,0.7,1], 'LineWidth', 2);
    end
end
legend([h1, h2, h3, h4], 'left cue', 'right cue', 'cells prefer left', 'cells prefer right', 'Location', 'southeast');
hold off;
xlabel('position');
ylabel('firing rate');
  

%% 再看看某个ca1所有trial的表征
% cell = 1;
% ca1_fr = squeeze(ca1_response(:, :, cell)); % tracklength*bs
% ymax = max(ca1_fr(:));
% left_trial = {};
% right_trial = {};
% figure;
% hold on;
% % 设置图例
% 
% for trial = 1:bs
%     if target(trial) == 0
%         % plot(1:100, ca1_fr(:, trial), '--', 'Color', [175,238,238]/255, 'LineWidth', 1);
%         right_trial{end+1} = ca1_fr(:, trial);
%     else
%         % plot(1:100, ca1_fr(:, trial), '--', 'Color', [255,218,185]/255, 'LineWidth', 1);
%         left_trial{end+1} = ca1_fr(:, trial);
%     end
% end
% left_average = reshape(cell2mat(left_trial), 100, []);
% left_fr_mean = mean(left_average, 2);
% left_fr_std = std(left_average, 0, 2);
% upper_bound = left_fr_mean+left_fr_std;
% lower_bound = left_fr_mean-left_fr_std;
% fill([1:100, fliplr(1:100)], [upper_bound', fliplr(lower_bound')], [1, 0.7, 0.7], 'EdgeColor', 'none');
% h1 = plot(1:100, left_fr_mean, 'r-', 'LineWidth', 2);
% 
% right_average = reshape(cell2mat(right_trial), 100, []);
% right_fr_mean = mean(right_average, 2);
% right_fr_std = std(right_average, 0, 2);
% upper_bound = right_fr_mean+right_fr_std;
% lower_bound = right_fr_mean-right_fr_std;
% fill([1:100, fliplr(1:100)], [upper_bound', fliplr(lower_bound')], [0.7, 0.7, 1], 'EdgeColor', 'none');
% h2 = plot(1:100, right_fr_mean, 'b-', 'LineWidth', 2);
% legend([h1, h2], 'left trial', 'right trial', 'Location', 'southeast');
% xlabel('position');
% ylabel('firing rate');

