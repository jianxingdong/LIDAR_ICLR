clear all; close all; clc;
num_Classs = 9;
target = load('C:\tmp\DNN\RGBLUV\test_targets_32_126513.mat');
target = target.data';
% img = load('F:\RGBLUV\tmp_32_train_test\test_digits_193379x6144.mat');
% img = load('C:\tmp\temp\test_digits_32_193379x6144.mat');
% img = img.data;
indx = load('C:\tmp\DNN\RGBLUV\test_index_32_var_126513.mat');
indx = indx.data;

% pred = load('F:\RGBLUV\ford_6ch_var2_32_193378x6144_nonorm_npyresults_conv101814\prediction_epoch299.mat');


% tag='ford_6ch_5var_32patch_9class_trts_seperate_conv20141127';
% tag ='ford_RGBLch_5var_32patch_9class_trts_seperate_conv20141127';
% tag ='ford_RLUVch_5var_32patch_9class_trts_seperate_conv20141127';
% tag ='ford_6ch_5var_32patch_9class_trts_seperate_numfilt323232_conv20141127';
% tag ='ford_6ch_5var_32patch_9class_trts_seperate_numfilt646464_conv20141127';
% tag ='ford_6ch_5var_32patch_9class_trts_seperate_filtw777_conv20141127';
tag ='ford_6ch_5var_32patch_9class_trts_seperate_filtw999_conv20141127';

pred = load(['C:\tmp\DNN\RGBLUV\' tag '\prediction_epoch99.mat']);


% pred = load('C:\tmp\DNN\RGBLUV\ford_6ch_5var_32patch_9class_trts_seperate_conv20141127\prediction_epoch99.mat');
% pred = load('C:\tmp\DNN\RGBLUV\\prediction_epoch99.mat');
% pred = load('C:\tmp\DNN\RGBLUV\\prediction_epoch99.mat');
% pred = load('C:\tmp\DNN\RGBLUV\\prediction_epoch99.mat');
% pred = load('C:\tmp\DNN\RGBLUV\\prediction_epoch99.mat');
% pred = load('C:\tmp\DNN\RGBLUV\\prediction_epoch99.mat');
% pred = load('C:\tmp\DNN\RGBLUV\\prediction_epoch99.mat');
% pred = load('C:\tmp\temp\noUV_prediction_epoch299.mat');
pred = pred.data';
res = max(pred');
[prob loc] = max(pred');
prob = prob';
label_predict= loc';
res = res';

img_info = [];
indx_info = [];
target_info = [];
predict_info = [];
for s = 1 : num_Classs
    loc = find(target(1:length(pred)) == s-1);
%     img_info = [img_info; img(loc,:)];
    indx_info = [indx_info; indx(loc,:)];
    target_info = [target_info; target(loc,:)];
%     predict_info = [predict_info; label_predict(loc,:)];
end

cf = zeros(num_Classs,num_Classs);
for i = 1 : length(target_info)
    cf(target_info(i)+1,label_predict(i)) = cf(target_info(i)+1,label_predict(i)) + 1;
end

for i = 1 : num_Classs
    cf(i,:) = cf(i,:)*100/sum(cf(i,:))
end

% x = [[ 58.86954521  22.91586778   8.09878058   6.41153463   3.7042718 ]
%  [ 14.14660328  49.45560497  22.1285079    9.78377549   4.48550836]
%  [  1.6825276   15.34069281  50.14084507  24.97145032   7.8644842 ]
%  [  0.46412539   0.84455604  10.46945142  59.24066043  28.98120673]
%  [  0.22829313   0.20546382   1.14907541  16.02617761  82.39099003]]

% x = [[ 50.05751975  32.65587852   7.2781655    4.35616228   5.65227395]
%  [ 10.27449778  58.34994633  18.84680264   6.57107806   5.9576752 ]
%  [  1.44651694  21.05062809  50.53673392  16.06395128  10.90216978]
%  [  0.28912729   0.84455604  15.36178955  42.64627558  40.85825154]
%  [  0.14458565   0.14458565   1.22517312  11.35377825  87.13187733]];

uni_scans = unique(indx_info(:,2));
count =0;
count_ = 0;
acc = [];
for s = 1 : length(uni_scans)
    loc = find(indx_info(:,2) == uni_scans(s));
    uni_img = unique(indx_info(loc,3));
%     if length(uni_img) > 1
%         count = count + length(uni_img);
%     end
%     count_ = count + 1;
    for i = 1 : length(uni_img)
        loc2 = find(indx_info(loc,3) == uni_img(i));
        final_loc = loc(loc2);
        cur_gt = target_info(final_loc,:)+1;
        cur_pred = label_predict(final_loc,:);
        for c = 1 : 9
            label_predict_ =[];
            loc3 = find(cur_gt == c);
            if isempty(loc)
                continue;
            else
                [t1 t2] = hist(cur_pred(loc3),1:num_Classs);
                label_predict_ = t2(find(t1==max(t1)))
            end
            if length(label_predict_) == 1
                acc = [acc; c label_predict_];
            end
        end
    end
end

C = confusionmat(acc(:,1),acc(:,2));
for i = 1 : num_Classs
    C(i,:) = C(i,:)*100/sum(C(i,:))
end
        
%%
close all;
fig1 = figure;
mat = C;           %# A 5-by-5 matrix of random values from 0 to 1
imagesc(mat);            %# Create a colored plot of the matrix values
colormap(flipud(gray));  %# Change the colormap to gray (so higher values are
                         %#   black and lower values are white)

textStrings = num2str(mat(:),'%0.2f');  %# Create strings from the matrix values
textStrings = strtrim(cellstr(textStrings));  %# Remove any space padding
[x,y] = meshgrid(1:9);   %# Create x and y coordinates for the strings
hStrings = text(x(:),y(:),textStrings(:),...      %# Plot the strings
                'HorizontalAlignment','center');
midValue = mean(get(gca,'CLim'));  %# Get the middle value of the color range
textColors = repmat(mat(:) > midValue,1,3);  %# Choose white or black for the
                                             %#   text color of the strings so
                                             %#   they can be easily seen over
                                             %#   the background color
set(hStrings,{'Color'},num2cell(textColors,2));  %# Change the text colors

set(gca,'XTick',1:9,...                         %# Change the axes tick marks
        'XTickLabel',{'Class-0','Class-1','Class-2','Class-3','Class-4','Class-5','Class-6','Class-7','Class-8'},...  %#   and tick labels
        'YTick',1:9,...
        'YTickLabel',{'Class-0','Class-1','Class-2','Class-3','Class-4','Class-5','Class-6','Class-7','Class-8'},...
        'TickLength',[0 0]);
title([tag '_' num2str(mean(diag(mat))) '%'],'interpreter','none');      
saveas(fig1,[tag '_image_level'],'png');
close all;
fig1 = figure;
mat = cf;           %# A 5-by-5 matrix of random values from 0 to 1
imagesc(mat);            %# Create a colored plot of the matrix values
colormap(flipud(gray));  %# Change the colormap to gray (so higher values are
                         %#   black and lower values are white)

textStrings = num2str(mat(:),'%0.2f');  %# Create strings from the matrix values
textStrings = strtrim(cellstr(textStrings));  %# Remove any space padding
[x,y] = meshgrid(1:9);   %# Create x and y coordinates for the strings
hStrings = text(x(:),y(:),textStrings(:),...      %# Plot the strings
                'HorizontalAlignment','center');
midValue = mean(get(gca,'CLim'));  %# Get the middle value of the color range
textColors = repmat(mat(:) > midValue,1,3);  %# Choose white or black for the
                                             %#   text color of the strings so
                                             %#   they can be easily seen over
                                             %#   the background color
set(hStrings,{'Color'},num2cell(textColors,2));  %# Change the text colors

set(gca,'XTick',1:9,...                         %# Change the axes tick marks
        'XTickLabel',{'Class-0','Class-1','Class-2','Class-3','Class-4','Class-5','Class-6','Class-7','Class-8'},...  %#   and tick labels
        'YTick',1:9,...
        'YTickLabel',{'Class-0','Class-1','Class-2','Class-3','Class-4','Class-5','Class-6','Class-7','Class-8'},...
        'TickLength',[0 0]);
title([tag '_' num2str(mean(diag(mat))) '%'],'interpreter','none');      
saveas(fig1,[tag '_patch_level'],'png');